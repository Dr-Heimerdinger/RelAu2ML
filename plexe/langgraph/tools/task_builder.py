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
