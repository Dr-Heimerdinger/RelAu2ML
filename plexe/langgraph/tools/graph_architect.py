from typing import Dict, Any, List, Optional
from langchain_core.tools import tool as langchain_tool

@langchain_tool
def validate_db_connection(connection_string: str) -> Dict[str, Any]:
    """
    Validate a database connection and retrieve available tables.
    
    Args:
        connection_string: Database connection string (e.g., postgresql://user:pass@host:port/db)
    
    Returns:
        Dictionary with connection status and available tables
    """
    from sqlalchemy import create_engine, inspect
    
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            table_info = {}
            for table in tables:
                columns = inspector.get_columns(table)
                pk_constraint = inspector.get_pk_constraint(table)
                fk_constraints = inspector.get_foreign_keys(table)
                
                table_info[table] = {
                    "columns": [{"name": c["name"], "type": str(c["type"])} for c in columns],
                    "primary_key": pk_constraint.get("constrained_columns", []),
                    "foreign_keys": [
                        {
                            "column": fk["constrained_columns"],
                            "references": f"{fk['referred_table']}.{fk['referred_columns']}"
                        }
                        for fk in fk_constraints
                    ]
                }
            
            return {
                "status": "connected",
                "tables": table_info,
                "table_count": len(tables)
            }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


@langchain_tool
def export_tables_to_csv(
    db_connection_string: str,
    output_dir: str,
    table_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Export database tables to CSV files with chunked writes for large tables.

    Args:
        db_connection_string: Database connection string
        output_dir: Directory to save CSV files
        table_names: Optional list of specific tables to export (exports all if None)

    Returns:
        Dictionary with export status and file paths
    """
    import pandas as pd
    from sqlalchemy import create_engine, inspect, text
    import os

    CHUNK_SIZE = 200_000

    os.makedirs(output_dir, exist_ok=True)

    try:
        engine = create_engine(db_connection_string)
        inspector = inspect(engine)

        if table_names is None:
            table_names = inspector.get_table_names()

        exported = []
        errors = []

        for table in table_names:
            file_path = os.path.join(output_dir, f"{table}.csv")
            try:
                with engine.connect() as conn:
                    row_count_result = conn.execute(
                        text(f'SELECT COUNT(*) FROM "{table}"')
                    )
                    total_rows = row_count_result.scalar()

                if total_rows <= CHUNK_SIZE:
                    df = pd.read_sql_table(table, engine)
                    df.to_csv(file_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
                else:
                    first_chunk = True
                    written_rows = 0
                    for chunk in pd.read_sql_table(table, engine, chunksize=CHUNK_SIZE):
                        chunk.to_csv(
                            file_path,
                            mode="w" if first_chunk else "a",
                            header=first_chunk,
                            index=False,
                            date_format="%Y-%m-%d %H:%M:%S",
                        )
                        written_rows += len(chunk)
                        first_chunk = False
                    total_rows = written_rows

                exported.append({"table": table, "path": file_path, "rows": total_rows})
            except Exception as e:
                errors.append({"table": table, "error": str(e)})

        return {
            "status": "success",
            "output_dir": output_dir,
            "exported_tables": exported,
            "errors": errors if errors else None,
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@langchain_tool
def extract_schema_metadata(db_connection_string: str) -> Dict[str, Any]:
    """
    Extract comprehensive schema metadata from a database.
    
    Args:
        db_connection_string: Database connection string
    
    Returns:
        Dictionary with tables, relationships, and temporal columns
    """
    from sqlalchemy import create_engine, inspect
    import pandas as pd
    
    try:
        engine = create_engine(db_connection_string)
        inspector = inspect(engine)
        
        tables = {}
        relationships = []
        temporal_columns = {}
        
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            pk_constraint = inspector.get_pk_constraint(table_name)
            fk_constraints = inspector.get_foreign_keys(table_name)
            
            table_cols = []
            for col in columns:
                col_type = str(col["type"]).lower()
                is_temporal = any(t in col_type for t in ['timestamp', 'date', 'time'])
                
                table_cols.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "is_temporal": is_temporal
                })
                
                if is_temporal:
                    if table_name not in temporal_columns:
                        temporal_columns[table_name] = []
                    temporal_columns[table_name].append(col["name"])
            
            tables[table_name] = {
                "columns": table_cols,
                "primary_key": pk_constraint.get("constrained_columns", []),
            }
            
            for fk in fk_constraints:
                relationships.append({
                    "source_table": table_name,
                    "source_column": fk["constrained_columns"][0] if fk["constrained_columns"] else None,
                    "target_table": fk["referred_table"],
                    "target_column": fk["referred_columns"][0] if fk["referred_columns"] else None,
                })
        
        return {
            "tables": tables,
            "relationships": relationships,
            "temporal_columns": temporal_columns
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
