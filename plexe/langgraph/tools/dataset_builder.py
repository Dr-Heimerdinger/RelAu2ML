from typing import Dict, Any
from langchain_core.tools import tool as langchain_tool

@langchain_tool
def get_csv_files_info(csv_dir: str) -> Dict[str, Any]:
    """
    Get information about CSV files in a directory.
    
    Args:
        csv_dir: Directory containing CSV files (relative or absolute path)
    
    Returns:
        Dictionary with file information
    """
    import os
    import pandas as pd
    
    # Convert to absolute path if needed
    csv_dir = os.path.abspath(csv_dir)
    
    # Check if directory exists
    if not os.path.exists(csv_dir):
        return {
            "error": f"Directory does not exist: {csv_dir}",
            "files": [],
            "count": 0
        }
    
    if not os.path.isdir(csv_dir):
        return {
            "error": f"Path is not a directory: {csv_dir}",
            "files": [],
            "count": 0
        }
    
    files = []
    try:
        for f in os.listdir(csv_dir):
            if f.endswith('.csv'):
                file_path = os.path.join(csv_dir, f)
                try:
                    df = pd.read_csv(file_path, nrows=1)
                    row_count = sum(1 for _ in open(file_path)) - 1
                    files.append({
                        "name": f.replace('.csv', ''),
                        "path": file_path,
                        "columns": list(df.columns),
                        "row_count": row_count
                    })
                except Exception as e:
                    files.append({"name": f, "error": str(e)})
    except Exception as e:
        return {
            "error": f"Error reading directory: {str(e)}",
            "files": [],
            "count": 0
        }
    
    return {"files": files, "count": len(files), "directory": csv_dir}


@langchain_tool
def get_temporal_statistics(csv_dir: str) -> Dict[str, Any]:
    """
    Analyze temporal columns in CSV files to determine val/test timestamps.
    
    Args:
        csv_dir: Directory containing CSV files (relative or absolute path)
    
    Returns:
        Dictionary with temporal analysis and suggested timestamps
    """
    import pandas as pd
    import os
    
    # Convert to absolute path if needed
    csv_dir = os.path.abspath(csv_dir)
    
    # Check if directory exists
    if not os.path.exists(csv_dir):
        return {
            "error": f"Directory does not exist: {csv_dir}",
            "temporal_stats": {},
            "suggested_splits": {}
        }
    
    temporal_stats = {}
    all_timestamps = []
    
    try:
        dir_files = os.listdir(csv_dir)
    except Exception as e:
        return {
            "error": f"Error reading directory: {str(e)}",
            "temporal_stats": {},
            "suggested_splits": {}
        }
    
    for f in dir_files:
        if not f.endswith('.csv'):
            continue
        
        table_name = f.replace('.csv', '')
        file_path = os.path.join(csv_dir, f)
        
        try:
            df = pd.read_csv(file_path)
            table_temporal = {}
            
            for col in df.columns:
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce', format='mixed')
                    valid_count = parsed.notna().sum()
                    if valid_count > len(df) * 0.5:
                        min_ts = parsed.min()
                        max_ts = parsed.max()
                        table_temporal[col] = {
                            "min": str(min_ts),
                            "max": str(max_ts),
                            "valid_count": int(valid_count)
                        }
                        all_timestamps.extend(parsed.dropna().tolist())
                except:
                    pass
            
            if table_temporal:
                temporal_stats[table_name] = table_temporal
        except Exception as e:
            temporal_stats[table_name] = {"error": str(e)}
    
    suggested_splits = {}
    if all_timestamps:
        # Use unique timestamps sorted by date for better split calculation
        unique_timestamps = sorted(list(set(all_timestamps)))
        n = len(unique_timestamps)
        
        if n > 2:
            val_ts = unique_timestamps[int(n * 0.7)]
            test_ts = unique_timestamps[int(n * 0.85)]
            
            # Ensure minimum gap of at least 30 days between val and test
            # This is important for tasks with timedelta requirements
            time_diff = test_ts - val_ts
            min_required_diff = pd.Timedelta(days=30)
            
            if time_diff < min_required_diff:
                # Adjust test timestamp to be at least 30 days after val
                test_ts = val_ts + min_required_diff
                # Make sure test_ts doesn't exceed max timestamp
                if test_ts > unique_timestamps[-1]:
                    # If we can't fit 30 days, adjust both timestamps
                    test_ts = unique_timestamps[-1]
                    val_ts = test_ts - min_required_diff
                    # Ensure val is not before min timestamp
                    if val_ts < unique_timestamps[0]:
                        val_ts = unique_timestamps[0]
                        test_ts = val_ts + min_required_diff
            
            suggested_splits = {
                "val_timestamp": str(val_ts),
                "test_timestamp": str(test_ts),
                "time_diff_days": str((test_ts - val_ts).days),
                "warning": None if time_diff >= min_required_diff else 
                    f"Adjusted timestamps to ensure minimum 30-day gap for timedelta requirements"
            }
        else:
            suggested_splits = {
                "error": "Not enough unique timestamps to create proper splits"
            }
    
    return {
        "temporal_stats": temporal_stats,
        "suggested_splits": suggested_splits
    }


@langchain_tool
def register_dataset_code(
    code: str,
    class_name: str,
    file_path: str
) -> Dict[str, str]:
    """
    Register generated Dataset class code.
    
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
        "code": code
    }
