from typing import Dict, Any
from langchain_core.tools import tool as langchain_tool

@langchain_tool
def analyze_csv_statistics(csv_dir: str) -> Dict[str, Any]:
    """
    Analyze statistical properties of CSV files.
    
    Args:
        csv_dir: Directory containing CSV files
    
    Returns:
        Dictionary with statistical analysis for each table
    """
    import pandas as pd
    import os
    
    stats = {}
    
    for f in os.listdir(csv_dir):
        if not f.endswith('.csv'):
            continue
        
        table_name = f.replace('.csv', '')
        file_path = os.path.join(csv_dir, f)
        
        try:
            df = pd.read_csv(file_path)
            
            table_stats = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": {},
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            }
            
            for col in df.columns:
                col_stats = {
                    "dtype": str(df[col].dtype),
                    "non_null_count": int(df[col].notna().sum()),
                    "null_count": int(df[col].isna().sum()),
                    "null_percentage": float(df[col].isna().sum() / len(df) * 100),
                    "unique_count": int(df[col].nunique()),
                }
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_stats["numeric_stats"] = {
                        "mean": float(df[col].mean()) if df[col].notna().any() else None,
                        "std": float(df[col].std()) if df[col].notna().any() else None,
                        "min": float(df[col].min()) if df[col].notna().any() else None,
                        "max": float(df[col].max()) if df[col].notna().any() else None,
                        "median": float(df[col].median()) if df[col].notna().any() else None,
                        "q25": float(df[col].quantile(0.25)) if df[col].notna().any() else None,
                        "q75": float(df[col].quantile(0.75)) if df[col].notna().any() else None,
                    }
                
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    top_values = df[col].value_counts().head(5)
                    col_stats["categorical_stats"] = {
                        "top_values": {str(k): int(v) for k, v in top_values.items()},
                        "is_high_cardinality": df[col].nunique() > 0.5 * len(df),
                    }
                
                table_stats["columns"][col] = col_stats
            
            stats[table_name] = table_stats
            
        except Exception as e:
            stats[table_name] = {"error": str(e)}
    
    return {
        "status": "success",
        "statistics": stats,
        "total_tables": len(stats)
    }


@langchain_tool
def detect_data_quality_issues(csv_dir: str) -> Dict[str, Any]:
    """
    Detect data quality issues in CSV files.
    
    Args:
        csv_dir: Directory containing CSV files
    
    Returns:
        Dictionary with data quality issues for each table
    """
    import pandas as pd
    import os
    
    issues = {}
    
    for f in os.listdir(csv_dir):
        if not f.endswith('.csv'):
            continue
        
        table_name = f.replace('.csv', '')
        file_path = os.path.join(csv_dir, f)
        
        try:
            df = pd.read_csv(file_path)
            table_issues = []
            
            for col in df.columns:
                null_pct = df[col].isna().sum() / len(df) * 100
                if null_pct > 50:
                    table_issues.append({
                        "severity": "high",
                        "column": col,
                        "issue": "high_missing_rate",
                        "details": f"{null_pct:.1f}% missing values"
                    })
                elif null_pct > 20:
                    table_issues.append({
                        "severity": "medium",
                        "column": col,
                        "issue": "moderate_missing_rate",
                        "details": f"{null_pct:.1f}% missing values"
                    })
                
                if df[col].dtype == 'object':
                    if df[col].nunique() == len(df):
                        table_issues.append({
                            "severity": "low",
                            "column": col,
                            "issue": "all_unique_values",
                            "details": "Every row has unique value (potential ID column)"
                        })
                    
                    if df[col].nunique() == 1:
                        table_issues.append({
                            "severity": "medium",
                            "column": col,
                            "issue": "constant_column",
                            "details": "All values are the same"
                        })
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    if (df[col] == 0).sum() / len(df) > 0.9:
                        table_issues.append({
                            "severity": "low",
                            "column": col,
                            "issue": "mostly_zeros",
                            "details": f"{(df[col] == 0).sum() / len(df) * 100:.1f}% zeros"
                        })
            
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                table_issues.append({
                    "severity": "medium",
                    "column": None,
                    "issue": "duplicate_rows",
                    "details": f"{duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)"
                })
            
            issues[table_name] = {
                "issues": table_issues,
                "issue_count": len(table_issues),
                "has_critical_issues": any(i["severity"] == "high" for i in table_issues)
            }
            
        except Exception as e:
            issues[table_name] = {"error": str(e)}
    
    return {
        "status": "success",
        "quality_issues": issues,
        "tables_with_issues": sum(1 for t in issues.values() if isinstance(t, dict) and t.get("issue_count", 0) > 0)
    }


@langchain_tool
def analyze_temporal_patterns(csv_dir: str) -> Dict[str, Any]:
    """
    Analyze temporal patterns in CSV files for time-series prediction tasks.
    
    Args:
        csv_dir: Directory containing CSV files
    
    Returns:
        Dictionary with temporal analysis
    """
    import pandas as pd
    import os
    
    temporal_analysis = {}
    all_timestamps = []
    
    for f in os.listdir(csv_dir):
        if not f.endswith('.csv'):
            continue
        
        table_name = f.replace('.csv', '')
        file_path = os.path.join(csv_dir, f)
        
        try:
            df = pd.read_csv(file_path)
            table_temporal = {"temporal_columns": {}}
            
            for col in df.columns:
                try:
                    # Specify format to avoid parsing warnings
                    parsed = pd.to_datetime(df[col], errors='coerce', format='mixed')
                    valid_count = parsed.notna().sum()
                    
                    if valid_count > len(df) * 0.5:
                        min_ts = parsed.min()
                        max_ts = parsed.max()
                        
                        time_range_days = (max_ts - min_ts).days if pd.notna(max_ts) and pd.notna(min_ts) else 0
                        
                        parsed_clean = parsed.dropna().sort_values()
                        if len(parsed_clean) > 1:
                            time_diffs = parsed_clean.diff().dropna()
                            avg_gap_hours = time_diffs.mean().total_seconds() / 3600 if not time_diffs.empty else 0
                        else:
                            avg_gap_hours = 0
                        
                        table_temporal["temporal_columns"][col] = {
                            "min": str(min_ts),
                            "max": str(max_ts),
                            "valid_count": int(valid_count),
                            "time_range_days": float(time_range_days),
                            "avg_gap_hours": float(avg_gap_hours),
                            "is_sorted": bool((parsed == parsed.sort_values()).all()),
                        }
                        
                        all_timestamps.extend(parsed_clean.tolist())
                except:
                    pass
            
            if table_temporal["temporal_columns"]:
                temporal_analysis[table_name] = table_temporal
                
        except Exception as e:
            temporal_analysis[table_name] = {"error": str(e)}
    
    suggested_splits = {}
    if all_timestamps:
        all_timestamps = sorted(all_timestamps)
        n = len(all_timestamps)
        suggested_splits = {
            "train_end": str(all_timestamps[int(n * 0.7)]),
            "val_end": str(all_timestamps[int(n * 0.85)]),
            "test_end": str(all_timestamps[-1]),
            "total_timestamps": n,
        }
    
    return {
        "status": "success",
        "temporal_analysis": temporal_analysis,
        "suggested_splits": suggested_splits,
        "has_temporal_data": len(temporal_analysis) > 0
    }


@langchain_tool
def analyze_table_relationships(csv_dir: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze relationships between tables based on schema and data.
    
    Args:
        csv_dir: Directory containing CSV files
        schema_info: Schema metadata with relationships
    
    Returns:
        Dictionary with relationship analysis
    """
    import pandas as pd
    import os
    
    relationship_analysis = {
        "foreign_key_stats": {},
        "join_recommendations": [],
        "dimension_fact_classification": {}
    }
    
    table_sizes = {}
    for f in os.listdir(csv_dir):
        if f.endswith('.csv'):
            table_name = f.replace('.csv', '')
            file_path = os.path.join(csv_dir, f)
            try:
                df = pd.read_csv(file_path)
                table_sizes[table_name] = len(df)
            except:
                table_sizes[table_name] = 0
    
    relationships = schema_info.get("relationships", [])
    for rel in relationships:
        source_table = rel.get("source_table")
        target_table = rel.get("target_table")
        source_column = rel.get("source_column")
        
        if not all([source_table, target_table, source_column]):
            continue
        
        try:
            source_file = os.path.join(csv_dir, f"{source_table}.csv")
            target_file = os.path.join(csv_dir, f"{target_table}.csv")
            
            if os.path.exists(source_file) and os.path.exists(target_file):
                source_df = pd.read_csv(source_file, usecols=[source_column] if source_column in pd.read_csv(source_file, nrows=0).columns else None)
                
                if source_column in source_df.columns:
                    fk_stats = {
                        "source_table": source_table,
                        "target_table": target_table,
                        "column": source_column,
                        "null_count": int(source_df[source_column].isna().sum()),
                        "null_percentage": float(source_df[source_column].isna().sum() / len(source_df) * 100),
                        "unique_count": int(source_df[source_column].nunique()),
                    }
                    
                    relationship_analysis["foreign_key_stats"][f"{source_table}.{source_column}"] = fk_stats
        except Exception as e:
            pass
    
    for table_name, size in table_sizes.items():
        has_fks = any(rel.get("source_table") == table_name for rel in relationships)
        is_referenced = any(rel.get("target_table") == table_name for rel in relationships)
        
        if has_fks and not is_referenced and size > 1000:
            classification = "fact"
        elif is_referenced and not has_fks:
            classification = "dimension"
        elif is_referenced and has_fks:
            classification = "dimension_with_hierarchy"
        else:
            classification = "standalone"
        
        relationship_analysis["dimension_fact_classification"][table_name] = {
            "classification": classification,
            "row_count": size,
            "has_foreign_keys": has_fks,
            "is_referenced": is_referenced,
        }
    
    return {
        "status": "success",
        "relationship_analysis": relationship_analysis
    }


@langchain_tool
def generate_eda_summary(
    statistics: Dict[str, Any],
    quality_issues: Dict[str, Any],
    temporal_analysis: Dict[str, Any],
    relationship_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate comprehensive EDA summary report.
    
    Args:
        statistics: Statistical analysis results
        quality_issues: Data quality issues
        temporal_analysis: Temporal pattern analysis
        relationship_analysis: Table relationship analysis
    
    Returns:
        Dictionary with comprehensive EDA summary
    """
    summary = {
        "overview": {},
        "key_findings": [],
        "recommendations": []
    }
    
    stats = statistics.get("statistics", {})
    total_rows = sum(t.get("row_count", 0) for t in stats.values() if isinstance(t, dict))
    total_columns = sum(t.get("column_count", 0) for t in stats.values() if isinstance(t, dict))
    
    summary["overview"] = {
        "total_tables": len(stats),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "has_temporal_data": temporal_analysis.get("has_temporal_data", False),
        "tables_with_quality_issues": quality_issues.get("tables_with_issues", 0),
    }
    
    if temporal_analysis.get("has_temporal_data"):
        summary["key_findings"].append({
            "category": "temporal",
            "finding": "Dataset contains temporal data suitable for time-series prediction",
            "details": f"Found {len(temporal_analysis.get('temporal_analysis', {}))} tables with temporal columns"
        })
        
        if temporal_analysis.get("suggested_splits"):
            summary["recommendations"].append({
                "category": "modeling",
                "recommendation": "Use temporal train/val/test splits",
                "details": temporal_analysis["suggested_splits"]
            })
    
    quality_issues_data = quality_issues.get("quality_issues", {})
    high_severity_count = sum(
        sum(1 for issue in t.get("issues", []) if issue.get("severity") == "high")
        for t in quality_issues_data.values() if isinstance(t, dict)
    )
    
    if high_severity_count > 0:
        summary["key_findings"].append({
            "category": "quality",
            "finding": f"Found {high_severity_count} high-severity data quality issues",
            "details": "Review tables with high missing rates before modeling"
        })
        
        summary["recommendations"].append({
            "category": "preprocessing",
            "recommendation": "Handle missing values in Dataset class",
            "details": "Consider imputation or dropping columns with >50% missing"
        })
    
    rel_analysis = relationship_analysis.get("relationship_analysis", {})
    dim_fact = rel_analysis.get("dimension_fact_classification", {})
    
    fact_tables = [t for t, info in dim_fact.items() if info.get("classification") == "fact"]
    dim_tables = [t for t, info in dim_fact.items() if info.get("classification") in ["dimension", "dimension_with_hierarchy"]]
    
    if fact_tables or dim_tables:
        summary["key_findings"].append({
            "category": "schema",
            "finding": f"Identified {len(fact_tables)} event/fact tables and {len(dim_tables)} entity/dimension tables",
            "details": {"event_tables": fact_tables, "entity_tables": dim_tables}
        })

        summary["recommendations"].append({
            "category": "modeling",
            "recommendation": (
                "Use entity/dimension tables (not event tables) as the prediction subject. "
                "Event tables provide historical context in SQL queries."
            ),
            "details": {
                "entity_tables_for_prediction": dim_tables,
                "event_tables_for_sql_context": fact_tables,
            }
        })
    
    return {
        "status": "success",
        "summary": summary
    }
