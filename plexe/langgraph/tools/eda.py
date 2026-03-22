from typing import Dict, Any
from langchain_core.tools import tool as langchain_tool


@langchain_tool
def analyze_all_csv(csv_dir: str, schema_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Perform full EDA on CSV files in a single pass using polars.

    Computes per-table statistics (row counts, null rates, cardinality, numeric
    summaries), detects data-quality issues, identifies temporal columns with
    suggested train/val/test splits, and classifies tables as fact/dimension.

    Args:
        csv_dir: Directory containing CSV files exported by the EDA agent.
        schema_info: Optional schema metadata with foreign-key relationships
                     (from extract_schema_metadata).  Used for table
                     classification and FK-stats.  Pass {} if unavailable.

    Returns:
        Dictionary with keys: statistics, quality_issues, temporal_analysis,
        suggested_splits, relationship_analysis, summary.
    """
    import os
    import polars as pl

    if schema_info is None:
        schema_info = {}

    csv_dir = os.path.abspath(csv_dir)
    if not os.path.isdir(csv_dir):
        return {"status": "error", "error": f"Directory not found: {csv_dir}"}

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        return {"status": "error", "error": f"No CSV files in {csv_dir}"}

    statistics: Dict[str, Any] = {}
    quality_issues: Dict[str, Any] = {}
    temporal_analysis: Dict[str, Any] = {}
    all_timestamps: list = []

    MIN_VALID_YEAR = 1900
    MAX_VALID_YEAR = 2100

    for fname in csv_files:
        table_name = fname.replace(".csv", "")
        file_path = os.path.join(csv_dir, fname)

        try:
            lf = pl.scan_csv(file_path, infer_schema_length=1000, try_parse_dates=True)
            schema = lf.collect_schema()
            col_names = schema.names()
            col_dtypes = {n: schema[n] for n in col_names}

            df_stats = lf.select(
                pl.count().alias("__row_count__"),
                *[pl.col(c).null_count().alias(f"__null__{c}") for c in col_names],
                *[pl.col(c).n_unique().alias(f"__nuniq__{c}") for c in col_names],
            ).collect()

            row_count = df_stats["__row_count__"][0]
            if row_count == 0:
                statistics[table_name] = {"row_count": 0, "column_count": len(col_names), "columns": {}}
                continue

            # Build numeric aggregations in one pass
            numeric_cols = [c for c, dt in col_dtypes.items() if dt.is_numeric()]
            num_aggs = {}
            if numeric_cols:
                num_df = lf.select(
                    *[pl.col(c).mean().alias(f"__mean__{c}") for c in numeric_cols],
                    *[pl.col(c).std().alias(f"__std__{c}") for c in numeric_cols],
                    *[pl.col(c).min().alias(f"__min__{c}") for c in numeric_cols],
                    *[pl.col(c).max().alias(f"__max__{c}") for c in numeric_cols],
                    *[pl.col(c).median().alias(f"__med__{c}") for c in numeric_cols],
                    *[pl.col(c).quantile(0.25).alias(f"__q25__{c}") for c in numeric_cols],
                    *[pl.col(c).quantile(0.75).alias(f"__q75__{c}") for c in numeric_cols],
                ).collect()
                for c in numeric_cols:
                    num_aggs[c] = {
                        "mean": _safe_float(num_df[f"__mean__{c}"][0]),
                        "std": _safe_float(num_df[f"__std__{c}"][0]),
                        "min": _safe_float(num_df[f"__min__{c}"][0]),
                        "max": _safe_float(num_df[f"__max__{c}"][0]),
                        "median": _safe_float(num_df[f"__med__{c}"][0]),
                        "q25": _safe_float(num_df[f"__q25__{c}"][0]),
                        "q75": _safe_float(num_df[f"__q75__{c}"][0]),
                    }

            # Per-column stats + quality issues
            col_stats = {}
            table_issues = []

            for c in col_names:
                null_count = df_stats[f"__null__{c}"][0]
                nunique = df_stats[f"__nuniq__{c}"][0]
                null_pct = null_count / row_count * 100 if row_count else 0.0

                cs = {
                    "dtype": str(col_dtypes[c]),
                    "non_null_count": row_count - null_count,
                    "null_count": null_count,
                    "null_percentage": round(null_pct, 2),
                    "unique_count": nunique,
                }

                if c in num_aggs:
                    cs["numeric_stats"] = num_aggs[c]

                if col_dtypes[c] == pl.Utf8 or col_dtypes[c] == pl.String:
                    cs["is_high_cardinality"] = nunique > 0.5 * row_count

                col_stats[c] = cs

                # Quality issues
                if null_pct > 50:
                    table_issues.append({"severity": "high", "column": c, "issue": "high_missing_rate", "details": f"{null_pct:.1f}% missing"})
                elif null_pct > 20:
                    table_issues.append({"severity": "medium", "column": c, "issue": "moderate_missing_rate", "details": f"{null_pct:.1f}% missing"})

                if str(col_dtypes[c]) in ("String", "Utf8"):
                    if nunique == row_count and row_count > 1:
                        table_issues.append({"severity": "low", "column": c, "issue": "all_unique_values", "details": "Potential ID column"})
                    elif nunique == 1:
                        table_issues.append({"severity": "medium", "column": c, "issue": "constant_column", "details": "All values identical"})

            statistics[table_name] = {
                "row_count": row_count,
                "column_count": len(col_names),
                "columns": col_stats,
            }

            quality_issues[table_name] = {
                "issues": table_issues,
                "issue_count": len(table_issues),
                "has_critical_issues": any(i["severity"] == "high" for i in table_issues),
            }

            # Temporal column detection
            table_temporal: Dict[str, Any] = {}

            # First check columns already typed as Date/Datetime by polars
            date_like_cols = [
                c for c, dt in col_dtypes.items()
                if dt in (pl.Date, pl.Datetime, pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("ms"))
                or str(dt).startswith("Datetime")
            ]

            for c in date_like_cols:
                try:
                    ts_stats = lf.select(
                        pl.col(c).min().alias("ts_min"),
                        pl.col(c).max().alias("ts_max"),
                        pl.col(c).drop_nulls().count().alias("ts_valid"),
                    ).collect()
                    min_ts = ts_stats["ts_min"][0]
                    max_ts = ts_stats["ts_max"][0]
                    valid_count = ts_stats["ts_valid"][0]

                    if min_ts is not None and max_ts is not None and valid_count > row_count * 0.5:
                        import datetime as _dt
                        if hasattr(min_ts, "year"):
                            if min_ts.year < MIN_VALID_YEAR or max_ts.year > MAX_VALID_YEAR:
                                continue
                        table_temporal[c] = {
                            "min": str(min_ts),
                            "max": str(max_ts),
                            "valid_count": int(valid_count),
                            "time_range_days": (max_ts - min_ts).days if hasattr(max_ts - min_ts, "days") else 0,
                        }
                        # Collect sample timestamps for split calculation (limit to avoid OOM)
                        sample_ts = (
                            lf.select(pl.col(c).drop_nulls().cast(pl.Datetime("us")))
                            .head(50000)
                            .collect()
                            .to_series()
                            .to_list()
                        )
                        all_timestamps.extend(sample_ts)
                except Exception:
                    pass

            # For string columns with datetime-like names, try parse a sample
            datetime_name_hints = ("date", "time", "created", "updated", "timestamp", "modified")
            for c in col_names:
                if c in date_like_cols or c in table_temporal:
                    continue
                c_lower = c.lower()
                if c_lower.endswith("id") or c_lower == "id":
                    continue
                if not any(h in c_lower for h in datetime_name_hints):
                    continue
                try:
                    sample = lf.select(pl.col(c).drop_nulls()).head(200).collect().to_series()
                    if len(sample) == 0:
                        continue
                    import pandas as _pd
                    parsed = _pd.to_datetime(sample.to_pandas(), errors="coerce", format="mixed")
                    valid = parsed.dropna()
                    if len(valid) < len(sample) * 0.5:
                        continue
                    min_ts = valid.min()
                    max_ts = valid.max()
                    if min_ts.year < MIN_VALID_YEAR or max_ts.year > MAX_VALID_YEAR:
                        continue
                    time_range_sec = (max_ts - min_ts).total_seconds()
                    if time_range_sec < 86400:
                        continue
                    table_temporal[c] = {
                        "min": str(min_ts),
                        "max": str(max_ts),
                        "valid_count": int(len(valid)),
                        "time_range_days": int((max_ts - min_ts).days),
                    }
                    # Collect full column timestamps via lazy scan for splits
                    try:
                        full_ts = _pd.to_datetime(
                            lf.select(pl.col(c).drop_nulls()).head(50000).collect().to_series().to_pandas(),
                            errors="coerce", format="mixed",
                        ).dropna()
                        all_timestamps.extend(full_ts.tolist())
                    except Exception:
                        pass
                except Exception:
                    pass

            if table_temporal:
                temporal_analysis[table_name] = {"temporal_columns": table_temporal}

        except Exception as e:
            statistics[table_name] = {"error": str(e)}
            quality_issues[table_name] = {"error": str(e)}

    # Suggested splits
    suggested_splits: Dict[str, Any] = {}
    if all_timestamps:
        import pandas as _pd
        unique_ts = sorted(set(all_timestamps))
        n = len(unique_ts)
        if n > 2:
            max_ts = unique_ts[-1]
            min_ts = unique_ts[0]
            data_range = _pd.Timestamp(max_ts) - _pd.Timestamp(min_ts)
            headroom = min(_pd.Timedelta(days=90), data_range * 0.15)
            min_gap = _pd.Timedelta(days=7)
            max_test_ts = _pd.Timestamp(max_ts) - headroom
            val_ts = _pd.Timestamp(unique_ts[int(n * 0.7)])
            test_ts = _pd.Timestamp(unique_ts[int(n * 0.85)])
            if test_ts > max_test_ts:
                test_ts = max_test_ts
            if test_ts - val_ts < min_gap:
                val_ts = test_ts - min_gap
                if val_ts < _pd.Timestamp(min_ts):
                    val_ts = _pd.Timestamp(min_ts)
            if test_ts > val_ts and test_ts > _pd.Timestamp(min_ts):
                suggested_splits = {
                    "train_end": str(val_ts),
                    "val_end": str(test_ts),
                    "test_end": str(max_ts),
                    "total_timestamps": n,
                    "headroom_days": str((_pd.Timestamp(max_ts) - test_ts).days),
                }

    # Table classification using schema relationships
    relationships = schema_info.get("relationships", [])
    table_sizes = {t: s.get("row_count", 0) for t, s in statistics.items() if isinstance(s, dict)}
    dim_fact: Dict[str, Any] = {}
    fk_stats: Dict[str, Any] = {}

    for table_name in table_sizes:
        has_fks = any(r.get("source_table") == table_name for r in relationships)
        is_referenced = any(r.get("target_table") == table_name for r in relationships)
        if has_fks and not is_referenced and table_sizes[table_name] > 1000:
            classification = "fact"
        elif is_referenced and not has_fks:
            classification = "dimension"
        elif is_referenced and has_fks:
            classification = "dimension_with_hierarchy"
        else:
            classification = "standalone"
        dim_fact[table_name] = {
            "classification": classification,
            "row_count": table_sizes[table_name],
            "has_foreign_keys": has_fks,
            "is_referenced": is_referenced,
        }

    # FK null stats (lightweight — only read the FK column)
    for rel in relationships:
        src_table = rel.get("source_table")
        src_col = rel.get("source_column")
        tgt_table = rel.get("target_table")
        if not src_table or not src_col:
            continue
        src_file = os.path.join(csv_dir, f"{src_table}.csv")
        if not os.path.exists(src_file):
            continue
        try:
            fk_lf = pl.scan_csv(src_file, infer_schema_length=500)
            if src_col in fk_lf.collect_schema().names():
                fk_agg = fk_lf.select(
                    pl.count().alias("total"),
                    pl.col(src_col).null_count().alias("nulls"),
                    pl.col(src_col).n_unique().alias("uniq"),
                ).collect()
                total = fk_agg["total"][0]
                nulls = fk_agg["nulls"][0]
                fk_stats[f"{src_table}.{src_col}"] = {
                    "source_table": src_table,
                    "target_table": tgt_table,
                    "column": src_col,
                    "null_count": nulls,
                    "null_percentage": round(nulls / max(total, 1) * 100, 2),
                    "unique_count": fk_agg["uniq"][0],
                }
        except Exception:
            pass

    relationship_analysis = {
        "foreign_key_stats": fk_stats,
        "dimension_fact_classification": dim_fact,
    }

    # Build summary
    total_rows = sum(s.get("row_count", 0) for s in statistics.values() if isinstance(s, dict))
    total_cols = sum(s.get("column_count", 0) for s in statistics.values() if isinstance(s, dict))
    fact_tables = [t for t, i in dim_fact.items() if i.get("classification") == "fact"]
    dim_tables = [t for t, i in dim_fact.items() if i.get("classification") in ("dimension", "dimension_with_hierarchy")]

    summary = {
        "overview": {
            "total_tables": len(statistics),
            "total_rows": total_rows,
            "total_columns": total_cols,
            "has_temporal_data": len(temporal_analysis) > 0,
            "tables_with_quality_issues": sum(1 for v in quality_issues.values() if isinstance(v, dict) and v.get("issue_count", 0) > 0),
        },
        "event_tables": fact_tables,
        "entity_tables": dim_tables,
    }

    return {
        "status": "success",
        "statistics": statistics,
        "quality_issues": quality_issues,
        "temporal_analysis": temporal_analysis,
        "suggested_splits": suggested_splits,
        "relationship_analysis": relationship_analysis,
        "summary": summary,
    }


def _safe_float(v) -> float:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
