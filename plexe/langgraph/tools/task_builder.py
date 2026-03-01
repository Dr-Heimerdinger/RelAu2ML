from typing import Dict, Any
from langchain_core.tools import tool as langchain_tool


@langchain_tool
def determine_lookback_window(
    csv_dir: str,
    event_table: str,
    entity_col: str,
    time_col: str,
    timedelta_days: int,
    task_description: str,
    entity_table: str = ""
) -> Dict[str, Any]:
    """
    Determine the correct lookback window and SQL pattern for the training table.
    MUST be called before designing the SQL query (Step 4 of the workflow).

    Analyzes the data to recommend one of five patterns:
    - Pattern A: Churn/absence -- cross join entity table + EXISTS with self.timedelta
    - Pattern B: Sparse events -- LEFT JOIN event table + WHERE IN with data-driven lookback
    - Pattern C: All-entity regression -- cross join entity table + COALESCE, no filter
    - Pattern D: Entity-creation filter -- LEFT JOIN entity ON creation_date <= timestamp
    - Link: Link prediction -- LEFT JOIN event table + LIST(DISTINCT)

    Args:
        csv_dir: Directory containing CSV files
        event_table: Name of the event/fact table CSV (without .csv extension)
        entity_col: Column name identifying the entity (e.g., 'customer_id')
        time_col: Column name for the temporal column in the event table (e.g., 't_dat')
        timedelta_days: The prediction window in days (e.g., 7 for weekly churn)
        task_description: Brief description of the task (e.g., 'predict customer churn')
        entity_table: Name of the entity/dimension table CSV (without .csv extension).
            If provided, the tool checks whether the entity table has a creation/start
            date column, which would indicate Pattern D.

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

        # --- Pattern D detection: entity table has a creation/start date ---
        entity_creation_col = None
        is_pattern_d_candidate = False
        if entity_table:
            entity_file = os.path.join(csv_dir, f"{entity_table}.csv")
            if os.path.exists(entity_file):
                try:
                    entity_df = pd.read_csv(entity_file, nrows=100)
                    creation_signals = [
                        'creation', 'created', 'start_date', 'publish',
                        'registered', 'signup', 'join_date', 'enrollment',
                    ]
                    for col in entity_df.columns:
                        col_lower = col.lower()
                        if col_lower.endswith('id') or col_lower == 'id':
                            continue
                        if any(sig in col_lower for sig in creation_signals):
                            # Verify it parses as a date
                            try:
                                parsed = pd.to_datetime(entity_df[col], errors='coerce')
                                if parsed.notna().sum() > len(entity_df) * 0.3:
                                    entity_creation_col = col
                                    is_pattern_d_candidate = True
                                    break
                            except Exception:
                                pass
                except Exception:
                    pass

        # --- Churn/absence detection (highest priority) ---
        churn_keywords = [
            'churn', 'no activity', 'no transaction', 'inactive',
            'absence', 'will not', "won't", 'stop', 'leave',
            'retain', 'retention', 'lapse', 'dormant',
        ]
        is_churn_task = any(kw in task_lower for kw in churn_keywords)

        # --- Link prediction detection ---
        link_keywords = [
            'list of', 'recommend', 'which items', 'purchase list',
            'link prediction', 'map@', 'precision@', 'recall@',
        ]
        is_link_task = any(kw in task_lower for kw in link_keywords)

        # --- All-entity regression detection (Pattern C) ---
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

        # --- Compute entity-level event frequency (used for A vs B decision) ---
        timedelta = pd.Timedelta(days=timedelta_days)
        date_range = df[time_col].max() - df[time_col].min()
        num_windows = max(date_range / timedelta, 1)

        entity_event_counts = df.groupby(entity_col).size()
        median_events_per_entity = entity_event_counts.median()
        median_events_per_window = median_events_per_entity / num_windows

        total_events = len(df)
        unique_entities = df[entity_col].nunique()
        events_per_day = total_events / max(date_range.days, 1)

        # ===================== Decision logic =====================

        if is_link_task:
            lookback = None
            pattern = "Link"
            reasoning = (
                f"LINK PREDICTION TASK DETECTED. Use the Link Prediction pattern: "
                f"LEFT JOIN event table on the forward window, GROUP BY source entity, "
                f"LIST(DISTINCT destination entity). No activity filter needed."
            )
        elif is_churn_task:
            lookback = "self.timedelta"
            pattern = "A"
            reasoning = (
                f"CHURN/ABSENCE TASK DETECTED. The lookback window MUST equal "
                f"self.timedelta (={timedelta_days} days) so that the 'previously active' "
                f"window is symmetric with the prediction window. Use Pattern A "
                f"(cross join entity table + EXISTS filter with self.timedelta)."
            )
        elif is_pattern_d_candidate and not is_all_entities_task:
            lookback = None
            pattern = "D"
            reasoning = (
                f"ENTITY-CREATION FILTER DETECTED. The entity table '{entity_table}' "
                f"has a creation/start date column '{entity_creation_col}'. "
                f"Use Pattern D: LEFT JOIN entity_table ON {entity_creation_col} <= t.timestamp, "
                f"then LEFT JOIN event table for the forward window. "
                f"This ensures only entities that existed before each timestamp are included."
            )
        elif is_all_entities_task:
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
            lookback = "self.timedelta"
            pattern = "A"
            reasoning = (
                f"FREQUENT ACTIVITY (non-churn): Median {median_events_per_window:.2f} events "
                f"per entity per {timedelta_days}-day window. Use Pattern A "
                f"(cross join entity table + EXISTS filter with self.timedelta)."
            )
        else:
            # Sparse activity: compute a data-driven lookback window.
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
                int(median_gap_days * 4),
                timedelta_days * 4,
                30,
            )
            raw_lookback_days = min(raw_lookback_days, int(date_range.days))

            if raw_lookback_days >= 548:
                lookback_str = "2 years"
            elif raw_lookback_days >= 274:
                lookback_str = "1 year"
            elif raw_lookback_days >= 137:
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
            "D": "LEFT JOIN entity table ON creation_date <= timestamp + LEFT JOIN events for forward window.",
            "Link": "LEFT JOIN event table + GROUP BY source entity + LIST(DISTINCT dest entity).",
        }

        return {
            "status": "success",
            "lookback_window": lookback,
            "recommended_pattern": pattern,
            "pattern_description": pattern_descriptions.get(pattern, ""),
            "reasoning": reasoning,
            "is_churn_task": is_churn_task,
            "is_all_entities_task": is_all_entities_task,
            "entity_creation_col": entity_creation_col,
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
        
        # Create a data-aware dummy timestamp_df for SQL validation.
        # At runtime, the real timestamp_df will be provided by the task framework.
        # Here we derive a representative date from the loaded tables so that
        # temporal JOINs find matching rows, producing a more realistic test.
        all_dates = []
        for f in os.listdir(csv_dir):
            if f.endswith('.csv'):
                try:
                    tbl = conn.execute(f"SELECT * FROM {f.replace('.csv', '')} LIMIT 0").fetchdf()
                    for col in tbl.columns:
                        col_lower = col.lower()
                        if col_lower.endswith('id') or col_lower == 'id':
                            continue
                        try:
                            sample = conn.execute(
                                f"SELECT DISTINCT \"{col}\" FROM {f.replace('.csv', '')} "
                                f"WHERE \"{col}\" IS NOT NULL LIMIT 500"
                            ).fetchdf()
                            parsed = pd.to_datetime(sample[col], errors='coerce').dropna()
                            if len(parsed) > len(sample) * 0.3:
                                all_dates.extend(parsed.tolist())
                        except Exception:
                            pass
                except Exception:
                    pass

        if all_dates:
            sorted_dates = sorted(all_dates)
            mid = sorted_dates[len(sorted_dates) // 2]
            base_ts = pd.Timestamp(mid).normalize()
        else:
            base_ts = pd.Timestamp('2020-01-01')

        timestamps_dummy = pd.DataFrame({
            'timestamp': pd.date_range(base_ts, periods=3, freq='7D')
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
    csv_dir: str,
    timedelta_days: int = 0
) -> Dict[str, Any]:
    """
    Validate that dataset timestamps are correctly set.

    Checks:
    1. val_timestamp and test_timestamp exist
    2. Timestamps are real dates within the data range (not Unix epoch)
    3. Gap between val and test is >= timedelta_days (prevents the ValueError
       'timedelta cannot be larger than the difference between val and test timestamps')

    Args:
        dataset_file_path: Path to dataset.py file
        csv_dir: Directory containing CSV files for temporal range check
        timedelta_days: The planned prediction window in days (e.g., 7, 30, 60).
            If provided, the tool verifies that test_timestamp - val_timestamp >= timedelta_days.
            This MUST be provided to prevent runtime ValueError.

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
            issues.append(
                f"Gap between val_timestamp and test_timestamp is only {time_diff} days. "
                f"Timestamps appear to be the same or inverted."
            )

        # Check gap >= timedelta (the root cause of the ValueError)
        if timedelta_days > 0 and time_diff < timedelta_days:
            issues.append(
                f"CRITICAL: Gap between val_timestamp and test_timestamp ({time_diff} days) "
                f"is smaller than the planned timedelta ({timedelta_days} days). "
                f"This will cause a ValueError at task initialization. "
                f"Fix: choose timestamps so that test_timestamp - val_timestamp >= {timedelta_days} days."
            )

        # Check against actual data range if CSV files available
        data_min_date = None
        data_max_date = None
        if csv_dir and os.path.exists(csv_dir):
            for f in os.listdir(csv_dir):
                if f.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(csv_dir, f))
                        for col in df.columns:
                            col_lower = col.lower()
                            if col_lower.endswith('id') or col_lower == 'id':
                                continue
                            try:
                                dates = pd.to_datetime(df[col], errors='coerce')
                                if dates.notna().sum() > len(df) * 0.5:
                                    col_min = dates.min()
                                    col_max = dates.max()
                                    if pd.notna(col_min) and col_min.year >= 1900:
                                        if data_min_date is None or col_min < data_min_date:
                                            data_min_date = col_min
                                    if pd.notna(col_max) and col_max.year <= 2100:
                                        if data_max_date is None or col_max > data_max_date:
                                            data_max_date = col_max
                            except Exception:
                                pass
                    except Exception:
                        pass

            if data_min_date and data_max_date:
                if val_timestamp < data_min_date or val_timestamp > data_max_date:
                    issues.append(
                        f"val_timestamp ({val_ts}) is outside the data range "
                        f"({data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')})"
                    )
                if test_timestamp < data_min_date or test_timestamp > data_max_date:
                    issues.append(
                        f"test_timestamp ({test_ts}) is outside the data range "
                        f"({data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')})"
                    )

        if issues:
            # Build a recommendation with suggested fix timestamps
            recommendation = (
                "Dataset timestamps must be fixed before creating tasks. "
                "Use fix_dataset_timestamps() to set timestamps where "
                f"test_timestamp - val_timestamp >= {max(timedelta_days, 1)} days "
                "and both timestamps fall within the data range."
            )
            if data_max_date and timedelta_days > 0:
                suggested_test = data_max_date - pd.Timedelta(days=max(timedelta_days, 7))
                suggested_val = suggested_test - pd.Timedelta(days=timedelta_days)
                recommendation += (
                    f" Suggested: val_timestamp='{suggested_val.strftime('%Y-%m-%d')}', "
                    f"test_timestamp='{suggested_test.strftime('%Y-%m-%d')}'."
                )

            return {
                "status": "invalid",
                "val_timestamp": val_ts,
                "test_timestamp": test_ts,
                "time_diff_days": time_diff,
                "timedelta_days": timedelta_days,
                "issues": issues,
                "recommendation": recommendation,
            }

        return {
            "status": "valid",
            "val_timestamp": val_ts,
            "test_timestamp": test_ts,
            "time_diff_days": time_diff,
            "timedelta_days": timedelta_days,
            "message": f"Dataset timestamps are valid (gap={time_diff}d >= timedelta={timedelta_days}d)"
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
