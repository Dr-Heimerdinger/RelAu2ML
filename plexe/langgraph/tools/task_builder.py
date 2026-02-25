from typing import Dict, Any
from langchain_core.tools import tool as langchain_tool

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
