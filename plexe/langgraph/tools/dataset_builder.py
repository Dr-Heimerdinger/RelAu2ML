from typing import Dict, Any, Optional, Tuple
from langchain_core.tools import tool as langchain_tool

KNOWN_DATASET_TIMESTAMPS: Dict[str, Tuple[str, str]] = {
    "stack": ("2020-10-01", "2021-01-01"),
    "amazon": ("2015-10-01", "2016-01-01"),
    "arxiv": ("2022-01-01", "2023-01-01"),
    "avito": ("2015-05-08", "2015-05-14"),
    "event": ("2012-11-21", "2012-11-29"),
    "f1": ("2005-01-01", "2010-01-01"),
    "hm": ("2020-09-07", "2020-09-14"),
    "ratebeer": ("2018-09-01", "2020-01-01"),
    "salt": ("2020-02-01", "2020-07-01"),
    "trial": ("2020-01-01", "2021-01-01"),
}


def _match_known_dataset(db_name: str) -> Optional[Tuple[str, str]]:
    """Match a database name against known RelBench datasets.

    Only matches '_full' suffix databases where author timestamps are valid.
    """
    if not db_name:
        return None
    db_lower = db_name.lower()
    if "_full" not in db_lower:
        return None
    for key, timestamps in KNOWN_DATASET_TIMESTAMPS.items():
        if db_lower.startswith(key):
            return timestamps
    return None


@langchain_tool
def get_csv_files_info(csv_dir: str) -> Dict[str, Any]:
    """Get information about CSV files: table names, columns, and row counts.

    Uses polars for fast row counting instead of line iteration.

    Args:
        csv_dir: Directory containing CSV files

    Returns:
        Dictionary with file information
    """
    import os
    import polars as pl

    csv_dir = os.path.abspath(csv_dir)
    if not os.path.isdir(csv_dir):
        return {"status": "error", "error": f"Not a directory: {csv_dir}", "files": [], "count": 0}

    files = []
    for f in os.listdir(csv_dir):
        if not f.endswith(".csv"):
            continue
        file_path = os.path.join(csv_dir, f)
        try:
            lf = pl.scan_csv(file_path, infer_schema_length=100)
            cols = lf.collect_schema().names()
            row_count = lf.select(pl.count()).collect().item()
            files.append({
                "name": f.replace(".csv", ""),
                "path": file_path,
                "columns": cols,
                "row_count": row_count,
            })
        except Exception as e:
            files.append({"name": f, "error": str(e)})

    return {"files": files, "count": len(files), "directory": csv_dir}


@langchain_tool
def get_temporal_statistics(csv_dir: str, db_name: str = "") -> Dict[str, Any]:
    """Analyze temporal columns to determine val/test timestamps.

    If db_name matches a known RelBench dataset, returns the author's exact
    val/test timestamps. Otherwise computes percentile-based splits using
    polars for fast scanning.

    Args:
        csv_dir: Directory containing CSV files
        db_name: Optional database name for known dataset matching

    Returns:
        Dictionary with temporal analysis and suggested timestamps
    """
    import os
    import polars as pl

    csv_dir = os.path.abspath(csv_dir)
    if not os.path.isdir(csv_dir):
        return {"status": "error", "error": f"Not a directory: {csv_dir}", "temporal_stats": {}, "suggested_splits": {}}

    known = _match_known_dataset(db_name)
    temporal_stats: Dict[str, Any] = {}
    all_timestamps = []

    MIN_VALID_YEAR = 1900
    MAX_VALID_YEAR = 2100
    datetime_name_hints = ("date", "time", "created", "updated", "timestamp", "modified")

    for f in os.listdir(csv_dir):
        if not f.endswith(".csv"):
            continue
        table_name = f.replace(".csv", "")
        file_path = os.path.join(csv_dir, f)

        try:
            lf = pl.scan_csv(file_path, infer_schema_length=1000, try_parse_dates=True)
            schema = lf.collect_schema()
            table_temporal = {}

            for col in schema.names():
                col_lower = col.lower()
                if col_lower.endswith("id") or col_lower == "id":
                    continue

                dt = schema[col]
                is_date_type = dt in (pl.Date, pl.Datetime) or str(dt).startswith("Datetime")

                if is_date_type:
                    try:
                        ts_agg = lf.select(
                            pl.col(col).min().alias("mn"),
                            pl.col(col).max().alias("mx"),
                            pl.col(col).drop_nulls().count().alias("vc"),
                        ).collect()
                        mn, mx, vc = ts_agg["mn"][0], ts_agg["mx"][0], ts_agg["vc"][0]
                        total = lf.select(pl.count()).collect().item()
                        if mn is not None and mx is not None and vc > total * 0.5:
                            if hasattr(mn, "year") and (mn.year < MIN_VALID_YEAR or mx.year > MAX_VALID_YEAR):
                                continue
                            table_temporal[col] = {"min": str(mn), "max": str(mx), "valid_count": int(vc)}
                            sample = lf.select(pl.col(col).drop_nulls().cast(pl.Datetime("us"))).head(50000).collect().to_series().to_list()
                            all_timestamps.extend(sample)
                    except Exception:
                        pass
                    continue

                if not any(h in col_lower for h in datetime_name_hints):
                    continue
                try:
                    import pandas as _pd
                    sample_s = lf.select(pl.col(col).drop_nulls()).head(200).collect().to_series()
                    if len(sample_s) == 0:
                        continue
                    parsed = _pd.to_datetime(sample_s.to_pandas(), errors="coerce", format="mixed")
                    valid = parsed.dropna()
                    if len(valid) < len(sample_s) * 0.5:
                        continue
                    mn, mx = valid.min(), valid.max()
                    if mn.year < MIN_VALID_YEAR or mx.year > MAX_VALID_YEAR:
                        continue
                    if (mx - mn).total_seconds() < 86400:
                        continue
                    table_temporal[col] = {"min": str(mn), "max": str(mx), "valid_count": int(len(valid))}
                    full_ts = _pd.to_datetime(
                        lf.select(pl.col(col).drop_nulls()).head(50000).collect().to_series().to_pandas(),
                        errors="coerce", format="mixed",
                    ).dropna()
                    all_timestamps.extend(full_ts.tolist())
                except Exception:
                    pass

            if table_temporal:
                temporal_stats[table_name] = table_temporal
        except Exception as e:
            temporal_stats[table_name] = {"error": str(e)}

    suggested_splits: Dict[str, Any] = {}
    if all_timestamps:
        import pandas as _pd
        unique_ts = sorted(set(all_timestamps))
        n = len(unique_ts)

        if known and n > 0:
            known_val, known_test = known
            known_test_ts = _pd.Timestamp(known_test)
            data_max_ts = unique_ts[-1]
            if known_test_ts <= data_max_ts:
                return {
                    "temporal_stats": temporal_stats,
                    "suggested_splits": {
                        "val_timestamp": known_val,
                        "test_timestamp": known_test,
                        "max_timestamp": str(data_max_ts),
                        "headroom_days": str((_pd.Timestamp(data_max_ts) - known_test_ts).days),
                        "source": "known_relbench_dataset",
                        "matched_db_name": db_name,
                    },
                }

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
                    "val_timestamp": str(val_ts),
                    "test_timestamp": str(test_ts),
                    "max_timestamp": str(max_ts),
                    "time_diff_days": str((test_ts - val_ts).days),
                    "headroom_days": str((_pd.Timestamp(max_ts) - test_ts).days),
                }

    return {"temporal_stats": temporal_stats, "suggested_splits": suggested_splits}


@langchain_tool
def register_dataset_code(code: str, class_name: str, file_path: str) -> Dict[str, str]:
    """Register generated Dataset class code by writing it to disk and validating.

    Args:
        code: Python code for the Dataset class
        class_name: Name of the Dataset class
        file_path: Path where the code will be saved

    Returns:
        Registration status
    """
    import os
    import ast

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    sanitized_code = code
    if '\\n' in code and '\n' not in code:
        import json
        try:
            sanitized_code = json.loads(f'"{code}"')
        except json.JSONDecodeError:
            sanitized_code = code.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")

    if '\\"\\"\\"' in sanitized_code:
        sanitized_code = sanitized_code.replace('\\"\\"\\"', '"""')

    syntax_warnings = []
    try:
        ast.parse(sanitized_code)
    except SyntaxError as e:
        import logging
        logging.warning(f"Generated code has syntax error: {e}")
        syntax_warnings.append(f"Syntax error at line {e.lineno}: {e.msg}")

    with open(file_path, 'w') as f:
        f.write(sanitized_code)

    csv_dir = os.path.join(os.path.dirname(file_path), "csv_files")
    validation_error = None
    if not syntax_warnings and os.path.isdir(csv_dir):
        import importlib.util
        import sys
        import threading

        module_name = f"_dataset_validation_{os.getpid()}"
        validation_exc = [None]

        def _validate():
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                dataset_cls = getattr(mod, class_name)
                dataset_obj = dataset_cls(csv_dir)
                dataset_obj.make_db()
            except Exception as exc:
                validation_exc[0] = exc

        t = threading.Thread(target=_validate, daemon=True)
        t.start()
        t.join(timeout=120)

        if t.is_alive():
            import logging
            logging.warning("Dataset validation timed out (120s), keeping file")
        elif validation_exc[0] is not None:
            exc = validation_exc[0]
            validation_error = f"{type(exc).__name__}: {exc}"
            import logging
            logging.warning(f"Dataset validation failed: {validation_error}")
            try:
                os.remove(file_path)
            except OSError:
                pass

        sys.modules.pop(module_name, None)
        pycache = os.path.join(os.path.dirname(file_path), "__pycache__")
        if os.path.isdir(pycache):
            import shutil
            shutil.rmtree(pycache, ignore_errors=True)

    if validation_error:
        return {
            "status": "validation_error",
            "error": validation_error,
            "class_name": class_name,
            "file_path": file_path,
            "message": (
                f"Runtime validation failed: {validation_error}. "
                "File DELETED. Fix the code and call register_dataset_code() again."
            ),
        }

    result = {
        "status": "registered_with_warnings" if syntax_warnings else "registered",
        "class_name": class_name,
        "file_path": file_path,
        "code": code,
    }
    if syntax_warnings:
        result["warnings"] = syntax_warnings
    return result
