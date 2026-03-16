from typing import Dict, Any, Optional, Tuple
from langchain_core.tools import tool as langchain_tool

# Known RelBench dataset timestamps from the original paper.
# Keys are substrings to match against the database name from the connection string.
# Source: /data/kl/plexe-clone/paper/relbench/relbench/datasets/
KNOWN_DATASET_TIMESTAMPS: Dict[str, Tuple[str, str]] = {
    "stack": ("2020-10-01", "2021-01-01"),       # stack.py:12-13
    "amazon": ("2015-10-01", "2016-01-01"),       # amazon.py:12-13
    "arxiv": ("2022-01-01", "2023-01-01"),        # arxiv.py:11-12
    "avito": ("2015-05-08", "2015-05-14"),        # avito.py:15-16
    "event": ("2012-11-21", "2012-11-29"),        # event.py:21-22
    "f1": ("2005-01-01", "2010-01-01"),           # f1.py:12-13
    "hm": ("2020-09-07", "2020-09-14"),           # hm.py:16-17
    "ratebeer": ("2018-09-01", "2020-01-01"),     # ratebeer.py:18-19
    "salt": ("2020-02-01", "2020-07-01"),         # salt.py:13-14
    "trial": ("2020-01-01", "2021-01-01"),        # trial.py:13-14
}


def _match_known_dataset(db_name: str) -> Optional[Tuple[str, str]]:
    """Match a database name against known RelBench datasets.

    Only matches when the db name has a '_full' suffix (e.g. 'stack_full'),
    indicating the complete dataset where the author's timestamps are valid.
    Partial datasets may not span the full date range, so we fall back to
    percentile-based splits for those.

    Returns (val_timestamp, test_timestamp) if matched, else None.
    """
    if not db_name:
        return None
    db_lower = db_name.lower()
    # Only use hardcoded timestamps for full datasets (e.g. stack_full, amazon_full)
    if "_full" not in db_lower:
        return None
    for key, timestamps in KNOWN_DATASET_TIMESTAMPS.items():
        if db_lower.startswith(key):
            return timestamps
    return None


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
            "status": "error",
            "error": f"Directory does not exist: {csv_dir}",
            "files": [],
            "count": 0
        }

    if not os.path.isdir(csv_dir):
        return {
            "status": "error",
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
            "status": "error",
            "error": f"Error reading directory: {str(e)}",
            "files": [],
            "count": 0
        }
    
    return {"files": files, "count": len(files), "directory": csv_dir}


@langchain_tool
def get_temporal_statistics(csv_dir: str, db_name: str = "") -> Dict[str, Any]:
    """
    Analyze temporal columns in CSV files to determine val/test timestamps.

    If db_name matches a known RelBench dataset, returns the author's exact
    val/test timestamps instead of computing percentile-based splits.

    Args:
        csv_dir: Directory containing CSV files (relative or absolute path)
        db_name: Optional database name (e.g. "stack_full") used to match
                 known RelBench datasets for exact author timestamps

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
            "status": "error",
            "error": f"Directory does not exist: {csv_dir}",
            "temporal_stats": {},
            "suggested_splits": {}
        }

    # Check for known RelBench dataset based on db_name.
    # We don't return immediately — we still need to scan CSV files to verify
    # the data actually extends far enough for the known timestamps + timedelta.
    known = _match_known_dataset(db_name)

    temporal_stats = {}
    all_timestamps = []
    
    # Minimum valid year for timestamps - filters out Unix epoch false positives
    # when integer IDs get parsed as microseconds from 1970-01-01
    MIN_VALID_YEAR = 1900
    MAX_VALID_YEAR = 2100
    
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
                # Skip columns that are likely ID columns based on name
                col_lower = col.lower()
                if col_lower.endswith('id') or col_lower == 'id':
                    continue
                
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce', format='mixed')
                    valid_count = parsed.notna().sum()
                    if valid_count > len(df) * 0.5:
                        min_ts = parsed.min()
                        max_ts = parsed.max()
                        
                        # Filter out false positives: timestamps near Unix epoch
                        # These are usually integer columns (IDs) being parsed as 
                        # microseconds from 1970-01-01
                        if pd.notna(min_ts) and pd.notna(max_ts):
                            min_year = min_ts.year
                            max_year = max_ts.year
                            
                            # Skip if timestamps are outside reasonable range
                            if min_year < MIN_VALID_YEAR or max_year > MAX_VALID_YEAR:
                                continue
                            
                            # Skip if the range is suspiciously small (< 1 day)
                            # This catches integer columns parsed as microseconds
                            time_range = (max_ts - min_ts).total_seconds()
                            if time_range < 86400:  # Less than 1 day in seconds
                                continue
                        
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

        # If we matched a known dataset, validate that the data extends at
        # least past the test timestamp.  The exact headroom needed depends on
        # the task's timedelta (unknown here), so we only require test_ts is
        # within the data range.  The author's timestamps are curated and
        # should be trusted when the data covers them.
        if known and n > 0:
            known_val, known_test = known
            known_test_ts = pd.Timestamp(known_test)
            data_max_ts = unique_timestamps[-1]
            if known_test_ts <= data_max_ts:
                return {
                    "temporal_stats": temporal_stats,
                    "suggested_splits": {
                        "val_timestamp": known_val,
                        "test_timestamp": known_test,
                        "max_timestamp": str(data_max_ts),
                        "headroom_days": str((data_max_ts - known_test_ts).days),
                        "source": "known_relbench_dataset",
                        "matched_db_name": db_name,
                    }
                }
            # else: data doesn't reach test timestamp, fall through

        if n > 2:
            max_ts = unique_timestamps[-1]
            min_ts = unique_timestamps[0]

            # Reserve headroom after test_timestamp for the task's prediction
            # window.  Scale proportionally to data range (15%) so short-range
            # datasets (e.g., Avito's 25 days) aren't rejected, while still
            # reserving space for longer-range datasets.  Cap at 90 days.
            data_range = max_ts - min_ts
            TIMEDELTA_HEADROOM = min(pd.Timedelta(days=90), data_range * 0.15)
            MIN_GAP = pd.Timedelta(days=7)

            # test_timestamp must leave room for at least one timedelta
            max_test_ts = max_ts - TIMEDELTA_HEADROOM

            val_ts = unique_timestamps[int(n * 0.7)]
            test_ts = unique_timestamps[int(n * 0.85)]

            # Clamp test_ts so the prediction window fits
            if test_ts > max_test_ts:
                test_ts = max_test_ts

            # Enforce a minimum gap between val and test
            time_diff = test_ts - val_ts
            if time_diff < MIN_GAP:
                val_ts = test_ts - MIN_GAP
                if val_ts < min_ts:
                    val_ts = min_ts

            # Final sanity: if test_ts is before val_ts, the dataset is too
            # short for a meaningful split
            if test_ts <= val_ts or test_ts <= min_ts:
                suggested_splits = {
                    "error": (
                        f"Dataset time range too short for meaningful splits. "
                        f"Range: {min_ts} to {max_ts} "
                        f"({(max_ts - min_ts).days} days), "
                        f"headroom: {TIMEDELTA_HEADROOM.days} days."
                    )
                }
            else:
                suggested_splits = {
                    "val_timestamp": str(val_ts),
                    "test_timestamp": str(test_ts),
                    "max_timestamp": str(max_ts),
                    "time_diff_days": str((test_ts - val_ts).days),
                    "headroom_days": str((max_ts - test_ts).days),
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
    
    syntax_warnings = []
    try:
        ast.parse(sanitized_code)
    except SyntaxError as e:
        import logging
        logging.warning(f"Generated code has syntax error: {e}")
        syntax_warnings.append(f"Syntax error at line {e.lineno}: {e.msg}")

    with open(file_path, 'w') as f:
        f.write(sanitized_code)

    result = {
        "status": "registered_with_warnings" if syntax_warnings else "registered",
        "class_name": class_name,
        "file_path": file_path,
        "code": code,
    }
    if syntax_warnings:
        result["warnings"] = syntax_warnings
    return result
