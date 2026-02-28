from typing import Dict, Any
from langchain_core.tools import tool as langchain_tool


@langchain_tool
def determine_lookback_window(
    csv_dir: str,
    event_table: str,
    entity_col: str,
    time_col: str,
    timedelta_days: int,
    task_description: str
) -> Dict[str, Any]:
    """
    Determine the correct lookback window for active entity filtering.
    MUST be called before designing the SQL query (Step 4 of the workflow).

    Analyzes the data to determine the correct lookback window and SQL pattern.
    For churn tasks the lookback is always self.timedelta (semantic requirement).
    For all-entity regression tasks no activity filter is used (Pattern C).
    For sparse-event tasks the lookback is computed from the actual median
    inter-event interval in the data and rounded to a clean SQL INTERVAL string.

    Args:
        csv_dir: Directory containing CSV files
        event_table: Name of the event/fact table CSV (without .csv extension)
        entity_col: Column name identifying the entity (e.g., 'customer_id')
        time_col: Column name for the temporal column (e.g., 't_dat')
        timedelta_days: The prediction window in days (e.g., 7 for weekly churn)
        task_description: Brief description of the task (e.g., 'predict customer churn')

    Returns:
        Recommended lookback window with evidence and SQL pattern to use
    """
    import os
    import pandas as pd

    try:
        csv_dir = os.path.abspath(csv_dir)
        file_path = os.path.join(csv_dir, f"{event_table}.csv")

        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"Event table CSV not found: {file_path}"
            }

        df = pd.read_csv(file_path)

        if entity_col not in df.columns:
            return {
                "status": "error",
                "error": f"Entity column '{entity_col}' not found in {event_table}.csv"
            }
        if time_col not in df.columns:
            return {
                "status": "error",
                "error": f"Time column '{time_col}' not found in {event_table}.csv"
            }

        # Parse time column
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])

        task_lower = task_description.lower()

        # Detect churn/absence/engagement tasks (semantic check — highest priority)
        churn_keywords = [
            'churn', 'no activity', 'no transaction', 'inactive',
            'absence', 'will not', "won't", 'stop', 'leave',
            'retain', 'retention', 'lapse', 'dormant',
        ]
        is_churn_task = any(kw in task_lower for kw in churn_keywords)

        # Detect "predict for ALL entities" tasks (Pattern C):
        # These are regression tasks where the entity is a catalog item (article,
        # product, post, study, ad) and the target is an aggregate value that can be 0.
        # Such tasks should NOT filter entities by recent activity.
        catalog_entity_signals = [
            'article', 'product', 'item', 'post', 'listing',
            'study', 'trial', 'facility', 'site', 'ad',
        ]
        aggregate_regression_signals = [
            'total', 'sum of', 'sales', 'revenue', 'ltv',
            'popularity', 'count of', 'clicks', 'votes', 'ctr',
            'how much', 'how many', 'number of', 'mae', 'rmse',
        ]
        active_qualifier_keywords = [
            'active', 'recently', 'recent ', 'engaged', 'retained',
            'previously purchased', 'previously active',
        ]
        entity_is_catalog = any(kw in entity_col.lower() for kw in catalog_entity_signals)
        is_aggregate_regression = any(kw in task_lower for kw in aggregate_regression_signals)
        has_active_qualifier = any(kw in task_lower for kw in active_qualifier_keywords)

        is_all_entities_task = (
            entity_is_catalog
            and is_aggregate_regression
            and not is_churn_task
            and not has_active_qualifier
        )

        # Compute entity-level event frequency (used for A vs B decision)
        timedelta = pd.Timedelta(days=timedelta_days)
        date_range = df[time_col].max() - df[time_col].min()
        num_windows = max(date_range / timedelta, 1)

        entity_event_counts = df.groupby(entity_col).size()
        median_events_per_entity = entity_event_counts.median()
        median_events_per_window = median_events_per_entity / num_windows

        total_events = len(df)
        unique_entities = df[entity_col].nunique()
        events_per_day = total_events / max(date_range.days, 1)

        # Decision logic
        if is_churn_task:
            # RULE: For churn/absence tasks, lookback MUST equal timedelta.
            # Churn means "was active in the preceding window, not active in the next."
            # Using a longer lookback would include stale entities and distort labels.
            lookback = "self.timedelta"
            pattern = "A"
            reasoning = (
                f"CHURN/ABSENCE TASK DETECTED. The lookback window MUST equal "
                f"self.timedelta (={timedelta_days} days) so that the 'previously active' "
                f"window is symmetric with the prediction window. Use Pattern A "
                f"(cross join entity table + EXISTS filter with self.timedelta)."
            )
        elif is_all_entities_task:
            # RULE: Regression tasks where the entity is a catalog item (article, product,
            # post, study, ad) and the target is an aggregate value that can be 0 should
            # include ALL entities without an activity filter.
            lookback = None
            pattern = "C"
            reasoning = (
                f"ALL-ENTITY REGRESSION TASK DETECTED (entity_col='{entity_col}', "
                f"catalog signals matched). A target of zero is semantically valid "
                f"(e.g., 0 sales means no sales this week, not missing data). "
                f"Do NOT filter entities by recent activity. Use Pattern C: "
                f"cross join entity table + LEFT JOIN events + COALESCE(SUM/COUNT, 0). "
                f"No WHERE EXISTS or WHERE IN clause for the entity population."
            )
        elif median_events_per_window >= 0.5:
            # Frequent activity (non-churn): use self.timedelta lookback
            lookback = "self.timedelta"
            pattern = "A"
            reasoning = (
                f"FREQUENT ACTIVITY (non-churn): Median {median_events_per_window:.2f} events "
                f"per entity per {timedelta_days}-day window. Use Pattern A "
                f"(cross join entity table + EXISTS filter with self.timedelta)."
            )
        else:
            # Sparse activity (non-churn): compute a data-driven lookback window.
            # Goal: the lookback must be wide enough that most active entities have
            # had at least a few events within it, even if events are infrequent.
            #
            # Strategy: compute the median inter-event gap for entities that have
            # more than one event, then set lookback = max(gap * 4, timedelta * 4).
            # Round the result to a human-readable SQL INTERVAL string.
            try:
                df_sorted = df[[entity_col, time_col]].sort_values(
                    [entity_col, time_col]
                )
                df_sorted = df_sorted.assign(
                    prev_time=df_sorted.groupby(entity_col)[time_col].shift(1)
                )
                gap_days_series = (
                    df_sorted[time_col] - df_sorted["prev_time"]
                ).dt.days.dropna()
                median_gap_days = float(gap_days_series.median()) if len(gap_days_series) else float(date_range.days)
            except Exception:
                median_gap_days = float(date_range.days)

            raw_lookback_days = max(
                int(median_gap_days * 4),   # cover ~4 median inter-event intervals
                timedelta_days * 4,          # or at least 4x the prediction window
                30,                          # floor: at least 30 days
            )
            # Cap at the actual data range so we never request more data than exists
            raw_lookback_days = min(raw_lookback_days, int(date_range.days))

            # Round to the nearest clean SQL INTERVAL string
            if raw_lookback_days >= 548:    # ≥ ~18 months
                lookback_str = "2 years"
            elif raw_lookback_days >= 274:  # ≥ ~9 months
                lookback_str = "1 year"
            elif raw_lookback_days >= 137:  # ≥ ~4.5 months
                lookback_str = "6 months"
            elif raw_lookback_days >= 60:
                lookback_str = "3 months"
            else:
                lookback_str = f"{raw_lookback_days} days"

            lookback = f"'{lookback_str}'"
            pattern = "B"
            reasoning = (
                f"SPARSE ACTIVITY (non-churn): Median {median_events_per_window:.2f} events "
                f"per entity per {timedelta_days}-day window. Median inter-event gap: "
                f"{median_gap_days:.0f} days. Computed lookback: {lookback_str} "
                f"(= max(gap*4={int(median_gap_days*4)}d, timedelta*4={timedelta_days*4}d), "
                f"rounded to clean interval, capped at data range). "
                f"Use Pattern B (LEFT JOIN event table + WHERE IN with {lookback} lookback)."
            )

        pattern_descriptions = {
            "A": "Cross join entity table + EXISTS filter with self.timedelta lookback.",
            "B": "LEFT JOIN event table + WHERE IN subquery with data-driven lookback.",
            "C": "Cross join entity table + LEFT JOIN events + COALESCE(agg, 0). No activity filter.",
        }

        return {
            "status": "success",
            "lookback_window": lookback,
            "recommended_pattern": pattern,
            "pattern_description": pattern_descriptions.get(pattern, ""),
            "reasoning": reasoning,
            "is_churn_task": is_churn_task,
            "is_all_entities_task": is_all_entities_task,
            "evidence": {
                "total_events": int(total_events),
                "unique_entities": int(unique_entities),
                "events_per_day": round(events_per_day, 2),
                "median_events_per_entity": round(float(median_events_per_entity), 2),
                "median_events_per_window": round(float(median_events_per_window), 4),
                "date_range_days": int(date_range.days),
                "timedelta_days": timedelta_days,
            },
            "must_use_lookback": lookback,
            "must_use_pattern": pattern,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@langchain_tool
def test_sql_query(
    csv_dir: str,
    query: str
) -> Dict[str, Any]:
    """
    Test a SQL query against CSV files using DuckDB.
    
    Args:
        csv_dir: Directory containing CSV files
        query: SQL query to test
    
    Returns:
        Query results or error
    """
    import duckdb
    import os
    import pandas as pd
    
    try:
        # Convert to absolute path to ensure files are found
        csv_dir = os.path.abspath(csv_dir)
        
        if not os.path.exists(csv_dir):
            return {
                "status": "error",
                "error": f"CSV directory does not exist: {csv_dir}"
            }
        
        conn = duckdb.connect(':memory:')
        
        # Load all CSV files as tables
        for f in os.listdir(csv_dir):
            if f.endswith('.csv'):
                table_name = f.replace('.csv', '')
                file_path = os.path.join(csv_dir, f)
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
        
        # Create a dummy timestamp_df for testing (will be provided by the task at runtime)
        # This is just for SQL validation
        timestamps_dummy = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=3, freq='D')
        })
        conn.register("timestamp_df", timestamps_dummy)
        
        result = conn.execute(query).fetchdf()
        
        return {
            "status": "success",
            "columns": list(result.columns),
            "row_count": len(result),
            "sample_data": result.head(10).to_dict(orient='records')
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@langchain_tool
def register_task_code(
    code: str,
    class_name: str,
    file_path: str,
    task_type: str
) -> Dict[str, str]:
    """
    Register generated Task class code.
    
    Args:
        code: Python code for the Task class
        class_name: Name of the Task class
        file_path: Full path where the code will be saved (e.g., workdir/session-xxx/task.py)
        task_type: Type of task (regression, binary_classification, multiclass_classification)
    
    Returns:
        Registration status
    """
    import os
    import ast
    
    # Normalize the file path
    file_path = os.path.normpath(os.path.abspath(file_path))
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Sanitize the code - handle escaped characters from JSON serialization
    sanitized_code = code
    
    # Check if the code has JSON-style escaping (e.g., \\n instead of real newlines)
    # This typically happens when LLM output gets double-serialized
    if '\\n' in code and '\n' not in code:
        # Looks like it's been JSON-escaped - unescape it
        import json
        try:
            # Wrap in quotes and parse as JSON string to unescape
            sanitized_code = json.loads(f'"{code}"')
        except json.JSONDecodeError:
            # If that fails, try manual unescaping of common sequences
            sanitized_code = code.replace('\\n', '\n')
            sanitized_code = sanitized_code.replace('\\t', '\t')
            sanitized_code = sanitized_code.replace('\\"', '"')
            sanitized_code = sanitized_code.replace("\\'", "'")
    
    # Additional fix: handle backslash-escaped triple quotes that break f-strings
    # Pattern: f\"\"\" should become f"""
    if '\\"\\"\\"' in sanitized_code:
        sanitized_code = sanitized_code.replace('\\"\\"\\"', '"""')
    
    # Validate that the code is syntactically valid Python
    try:
        ast.parse(sanitized_code)
    except SyntaxError as e:
        # If there's still a syntax error, log it but continue
        import logging
        logging.warning(f"Generated code has syntax error: {e}")
    
    with open(file_path, 'w') as f:
        f.write(sanitized_code)
    
    return {
        "status": "registered",
        "class_name": class_name,
        "file_path": file_path,
        "task_type": task_type,
        "code": code
    }


@langchain_tool
def validate_dataset_timestamps(
    dataset_file_path: str,
    csv_dir: str
) -> Dict[str, Any]:
    """
    Validate that dataset timestamps are correctly set.
    
    Checks:
    1. val_timestamp and test_timestamp exist
    2. Timestamps are real dates within the data range (not Unix epoch)
    3. Sufficient gap between val and test (at least 30 days for timedelta)
    
    Args:
        dataset_file_path: Path to dataset.py file
        csv_dir: Directory containing CSV files for temporal range check
    
    Returns:
        Validation results with status and any issues found
    """
    import os
    import ast
    import pandas as pd
    
    try:
        # Read the dataset file
        if not os.path.exists(dataset_file_path):
            return {
                "status": "error",
                "error": f"Dataset file not found: {dataset_file_path}"
            }
        
        with open(dataset_file_path, 'r') as f:
            dataset_code = f.read()
        
        # Parse to find val_timestamp and test_timestamp
        tree = ast.parse(dataset_code)
        val_ts = None
        test_ts = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == 'val_timestamp':
                            # Extract the timestamp string
                            if isinstance(node.value, ast.Call):
                                if len(node.value.args) > 0:
                                    if isinstance(node.value.args[0], ast.Constant):
                                        val_ts = node.value.args[0].value
                        elif target.id == 'test_timestamp':
                            if isinstance(node.value, ast.Call):
                                if len(node.value.args) > 0:
                                    if isinstance(node.value.args[0], ast.Constant):
                                        test_ts = node.value.args[0].value
        
        if not val_ts or not test_ts:
            return {
                "status": "error",
                "error": "Could not find val_timestamp or test_timestamp in dataset.py"
            }
        
        # Parse timestamps
        val_timestamp = pd.Timestamp(val_ts)
        test_timestamp = pd.Timestamp(test_ts)
        
        issues = []
        
        # Check if timestamps are suspiciously close to Unix epoch (1970-01-01)
        epoch = pd.Timestamp("1970-01-01")
        if abs((val_timestamp - epoch).days) < 365:
            issues.append(f"val_timestamp ({val_ts}) is suspiciously close to Unix epoch (1970-01-01)")
        if abs((test_timestamp - epoch).days) < 365:
            issues.append(f"test_timestamp ({test_ts}) is suspiciously close to Unix epoch (1970-01-01)")
        
        # Check gap between val and test
        time_diff = (test_timestamp - val_timestamp).days
        if time_diff < 1:
            # Only flag as issue if timestamps are essentially the same day
            # The actual required gap depends on the task's timedelta (e.g., 4 days, 7 days, 30 days)
            issues.append(
                f"Gap between val_timestamp and test_timestamp is only {time_diff} days. "
                f"Timestamps appear to be the same or very close - this is likely an error."
            )
        
        # Check against actual data range if CSV files available
        if csv_dir and os.path.exists(csv_dir):
            min_date = None
            max_date = None
            
            for f in os.listdir(csv_dir):
                if f.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(csv_dir, f))
                        for col in df.columns:
                            try:
                                dates = pd.to_datetime(df[col], errors='coerce')
                                if dates.notna().sum() > len(df) * 0.5:
                                    col_min = dates.min()
                                    col_max = dates.max()
                                    if min_date is None or col_min < min_date:
                                        min_date = col_min
                                    if max_date is None or col_max > max_date:
                                        max_date = col_max
                            except:
                                pass
                    except:
                        pass
            
            if min_date and max_date:
                if val_timestamp < min_date or val_timestamp > max_date:
                    issues.append(
                        f"val_timestamp ({val_ts}) is outside the data range "
                        f"({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})"
                    )
                if test_timestamp < min_date or test_timestamp > max_date:
                    issues.append(
                        f"test_timestamp ({test_ts}) is outside the data range "
                        f"({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})"
                    )
        
        if issues:
            return {
                "status": "invalid",
                "val_timestamp": val_ts,
                "test_timestamp": test_ts,
                "time_diff_days": time_diff,
                "issues": issues,
                "recommendation": "Dataset timestamps must be fixed before creating tasks. "
                                 "Use proper dates within the data range with at least 30 days gap."
            }
        
        return {
            "status": "valid",
            "val_timestamp": val_ts,
            "test_timestamp": test_ts,
            "time_diff_days": time_diff,
            "message": "Dataset timestamps are valid"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@langchain_tool
def fix_dataset_timestamps(
    dataset_file_path: str,
    val_timestamp: str,
    test_timestamp: str
) -> Dict[str, Any]:
    """
    Fix the val_timestamp and test_timestamp in an existing dataset.py file.
    
    Use this tool when validate_dataset_timestamps returns invalid status.
    The tool will update the timestamps in the dataset.py file.
    
    Args:
        dataset_file_path: Path to dataset.py file
        val_timestamp: New val_timestamp in YYYY-MM-DD format
        test_timestamp: New test_timestamp in YYYY-MM-DD format (must be after val_timestamp)
    
    Returns:
        Status of the fix operation
    """
    import os
    import re
    import pandas as pd
    
    try:
        # Validate timestamps
        val_ts = pd.Timestamp(val_timestamp)
        test_ts = pd.Timestamp(test_timestamp)
        
        if test_ts <= val_ts:
            return {
                "status": "error",
                "error": "test_timestamp must be after val_timestamp"
            }
        
        # Note: We don't enforce a minimum gap here because different tasks
        # have different timedelta requirements (e.g., 7 days for weekly churn,
        # 30 days for monthly predictions). The gap should be >= task's timedelta.
        
        # Read the dataset file
        if not os.path.exists(dataset_file_path):
            return {
                "status": "error",
                "error": f"Dataset file not found: {dataset_file_path}"
            }
        
        with open(dataset_file_path, 'r') as f:
            content = f.read()
        
        # Replace val_timestamp
        val_pattern = r'val_timestamp\s*=\s*pd\.Timestamp\(["\'][^"\']*["\']\)'
        val_replacement = f'val_timestamp = pd.Timestamp("{val_timestamp}")'
        new_content = re.sub(val_pattern, val_replacement, content)
        
        # Replace test_timestamp
        test_pattern = r'test_timestamp\s*=\s*pd\.Timestamp\(["\'][^"\']*["\']\)'
        test_replacement = f'test_timestamp = pd.Timestamp("{test_timestamp}")'
        new_content = re.sub(test_pattern, test_replacement, new_content)
        
        # Check if replacements were made
        if new_content == content:
            return {
                "status": "error",
                "error": "Could not find val_timestamp or test_timestamp patterns in dataset.py"
            }
        
        # Write the updated content
        with open(dataset_file_path, 'w') as f:
            f.write(new_content)
        
        return {
            "status": "success",
            "message": f"Updated timestamps in {dataset_file_path}",
            "val_timestamp": val_timestamp,
            "test_timestamp": test_timestamp,
            "time_diff_days": (test_ts - val_ts).days
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
