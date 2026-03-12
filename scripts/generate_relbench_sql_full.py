#!/usr/bin/env python3
"""
Generate SQL DDL and export full CSV from RelBench datasets (no sampling).
Table names are NOT suffixed; the database itself carries the _full suffix.
Support any RelBench dataset: rel-f1, rel-amazon, rel-hm, rel-stack, etc.
"""

import os
import sys
import csv
import argparse
import pandas as pd
from relbench.datasets import get_dataset
from pathlib import Path


def pandas_dtype_to_sql(dtype, col_name):
    """Convert pandas dtype to PostgreSQL type.

    Uses BIGINT for all integer types (pandas Int64 is 64-bit).
    Uses DOUBLE PRECISION / REAL matching the pandas float width.
    Uses TIMESTAMP for any datetime type.
    """
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(dtype):
        # Preserve float32 vs float64 precision
        return "REAL" if dtype == 'float32' else "DOUBLE PRECISION"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"


def q(name):
    """Quote a SQL identifier with double quotes."""
    return f'"{name}"'


def get_primary_key_column(table):
    """Get primary key column from table"""
    if hasattr(table, 'pkey_col') and table.pkey_col:
        return table.pkey_col
    return None


def full_table_name(table_name):
    """Return table name (no _full suffix; the database carries the suffix)"""
    return table_name


def generate_create_table_sql(table_name, table):
    """Generate SQL CREATE TABLE statement for PostgreSQL"""
    tname = full_table_name(table_name)
    sql = f"CREATE TABLE {tname} (\n"

    columns = []
    for col_name in table.df.columns:
        dtype = table.df[col_name].dtype
        sql_type = pandas_dtype_to_sql(dtype, col_name)
        col_def = f"    {q(col_name)} {sql_type}"
        columns.append(col_def)

    sql += ",\n".join(columns)

    pkey = get_primary_key_column(table)
    if pkey:
        sql += f",\n    PRIMARY KEY ({q(pkey)})"

    sql += "\n);"
    return sql


def generate_temp_table_sql(table_name, table):
    """Generate temporary table with all columns as TEXT"""
    tname = full_table_name(table_name)
    sql = f"CREATE TEMP TABLE temp_{tname} (\n"
    columns = [f"    {q(col)} TEXT" for col in table.df.columns]
    sql += ",\n".join(columns)
    sql += "\n);"
    return sql


def generate_insert_sql(table_name, table):
    """Generate INSERT statement with type conversion"""
    tname = full_table_name(table_name)
    columns = []
    conversions = []

    for col_name in table.df.columns:
        dtype = table.df[col_name].dtype
        columns.append(q(col_name))

        if pd.api.types.is_integer_dtype(dtype):
            conversions.append(f"    NULLIF({q(col_name)}, '')::BIGINT")
        elif pd.api.types.is_float_dtype(dtype):
            pg_type = "REAL" if dtype == 'float32' else "DOUBLE PRECISION"
            conversions.append(f"    NULLIF({q(col_name)}, '')::{pg_type}")
        elif pd.api.types.is_bool_dtype(dtype):
            conversions.append(f"    NULLIF({q(col_name)}, '')::BOOLEAN")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            conversions.append(f"    NULLIF({q(col_name)}, '')::TIMESTAMP")
        else:
            conversions.append(f"    NULLIF({q(col_name)}, '')")

    col_list = ', '.join(columns)
    sql = f"INSERT INTO {tname} ({col_list})\nSELECT \n"
    sql += ",\n".join(conversions)
    sql += f"\nFROM temp_{tname};"
    return sql


def generate_foreign_keys_sql(db):
    """Generate ALTER TABLE statements for foreign keys"""
    fk_statements = []

    for table_name, table in db.table_dict.items():
        tname = full_table_name(table_name)
        if hasattr(table, 'fkey_col_to_pkey_table'):
            for fkey_col, ref_table in table.fkey_col_to_pkey_table.items():
                if ref_table in db.table_dict:
                    ref_tname = full_table_name(ref_table)
                    ref_pkey = get_primary_key_column(db.table_dict[ref_table])
                    if ref_pkey:
                        # Sanitize constraint name (remove special chars)
                        safe_fkey = fkey_col.replace(' ', '_').replace(':', '_')
                        fk_name = f"fk_{tname}_{safe_fkey}"
                        stmt = f"ALTER TABLE {tname} ADD CONSTRAINT {fk_name}\n"
                        stmt += f"    FOREIGN KEY ({q(fkey_col)}) REFERENCES {ref_tname}({q(ref_pkey)});"
                        fk_statements.append(stmt)

    return fk_statements


def generate_indexes_sql(db):
    """Generate indexes for foreign keys and time columns"""
    index_statements = []

    for table_name, table in db.table_dict.items():
        tname = full_table_name(table_name)
        if hasattr(table, 'fkey_col_to_pkey_table'):
            for fkey_col in table.fkey_col_to_pkey_table.keys():
                safe_fkey = fkey_col.replace(' ', '_').replace(':', '_')
                idx_name = f"idx_{tname}_{safe_fkey}"
                stmt = f"CREATE INDEX {idx_name} ON {tname}({q(fkey_col)}) WHERE {q(fkey_col)} IS NOT NULL;"
                index_statements.append(stmt)

        if hasattr(table, 'time_col') and table.time_col:
            safe_time = table.time_col.replace(' ', '_').replace(':', '_')
            idx_name = f"idx_{tname}_{safe_time}"
            stmt = f"CREATE INDEX {idx_name} ON {tname}({q(table.time_col)});"
            index_statements.append(stmt)

    return index_statements


def generate_complete_sql(db, dataset_name):
    """Generate complete SQL import script for full dataset"""
    sql_parts = []

    sql_parts.append(f"-- RelBench {dataset_name.upper()} Database Schema (FULL - no sampling)")
    sql_parts.append("-- Auto-generated from RelBench dataset")
    sql_parts.append(f"-- Total tables: {len(db.table_dict)}")
    sql_parts.append("-- Table names do NOT carry a _full suffix (the database name does)")
    sql_parts.append("")

    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 1: Drop existing tables'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")

    for table_name in reversed(list(db.table_dict.keys())):
        sql_parts.append(f"DROP TABLE IF EXISTS {full_table_name(table_name)} CASCADE;")
    sql_parts.append("")
    sql_parts.append("\\echo 'Tables dropped'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")

    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 2: Create tables'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")

    for table_name, table in db.table_dict.items():
        sql_parts.append(generate_create_table_sql(table_name, table))
        sql_parts.append("")

    sql_parts.append("\\echo 'Tables created'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")

    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 3: Create temp tables for import'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")

    for table_name, table in db.table_dict.items():
        sql_parts.append(generate_temp_table_sql(table_name, table))
        sql_parts.append("")

    sql_parts.append("\\echo 'Temp tables created'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")

    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 4: Import CSV into temp tables'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")

    for table_name in db.table_dict.keys():
        tname = full_table_name(table_name)
        sql_parts.append(f"\\echo '   Importing {table_name}...'")
        sql_parts.append(f"\\copy temp_{tname} FROM '/tmp/{table_name}.csv' WITH (FORMAT CSV, HEADER, DELIMITER ',', QUOTE '\"');")
        sql_parts.append("")

    sql_parts.append("\\echo 'CSV imported to temp tables'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")

    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 5: Transfer data with type conversion'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")

    for table_name, table in db.table_dict.items():
        tname = full_table_name(table_name)
        sql_parts.append(f"\\echo '   Processing {tname}...'")
        sql_parts.append(generate_insert_sql(table_name, table))
        sql_parts.append("")

    sql_parts.append("\\echo 'Data transferred with NULL handling'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")

    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 6: Add Foreign Keys'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")

    fk_statements = generate_foreign_keys_sql(db)
    for stmt in fk_statements:
        sql_parts.append(stmt)
        sql_parts.append("")

    sql_parts.append("\\echo 'Foreign keys added'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")

    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 7: Create Indexes'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")

    index_statements = generate_indexes_sql(db)
    for stmt in index_statements:
        sql_parts.append(stmt)
    sql_parts.append("")

    sql_parts.append("\\echo 'Indexes created'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")

    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Summary'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")
    sql_parts.append("SELECT ")
    sql_parts.append("    table_name,")
    sql_parts.append("    to_char(record_count, 'FM999,999,999') as records")
    sql_parts.append("FROM (")

    union_parts = []
    for table_name in db.table_dict.keys():
        tname = full_table_name(table_name)
        union_parts.append(f"    SELECT '{tname}' as table_name, COUNT(*) as record_count FROM {tname}")

    sql_parts.append("\n    UNION ALL\n".join(union_parts))
    sql_parts.append(") t")
    sql_parts.append("ORDER BY record_count DESC;")
    sql_parts.append("")

    sql_parts.append("\\echo ''")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'FULL IMPORT COMPLETE'")
    sql_parts.append("\\echo '========================================='")

    return "\n".join(sql_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SQL DDL and export FULL CSV from RelBench datasets (no sampling). '
                    'Tables are stored without a _full suffix; the database name carries the suffix.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s rel-f1
  %(prog)s rel-amazon --output-dir ./amazon_full_data
  %(prog)s rel-hm --no-download

Supported datasets:
  rel-f1, rel-amazon, rel-hm, rel-stack, rel-trial,
  rel-event, rel-avito, rel-salt, rel-arxiv, rel-ratebeer
        '''
    )

    parser.add_argument('dataset', type=str,
                        help='Dataset name (e.g., rel-f1, rel-amazon)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for CSV and SQL files (default: ./{dataset}_full_data)')
    parser.add_argument('--no-download', action='store_true',
                        help='Skip download if dataset already exists in cache')

    args = parser.parse_args()

    dataset_name = args.dataset

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dataset_short = dataset_name.replace('rel-', '')
        output_dir = Path(f"./{dataset_short}_full_data")

    print("=" * 60)
    print(f"RelBench {dataset_name.upper()} - Full SQL Generation (no sampling)")
    print("=" * 60)
    print()

    print(f"Downloading {dataset_name} dataset...")
    try:
        dataset = get_dataset(dataset_name, download=not args.no_download)
        db = dataset.get_db(upto_test_timestamp=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"\nMake sure '{dataset_name}' is a valid RelBench dataset.")
        sys.exit(1)

    print(f"Dataset loaded: {len(db.table_dict)} tables")
    print()

    output_dir.mkdir(exist_ok=True)

    import numpy as np

    def clean_value(x):
        """Convert any value to a clean string safe for CSV export.

        Only strips NUL bytes (\x00) which break the C-level CSV writer.
        Newlines/carriage-returns are intentionally kept as-is because
        QUOTE_ALL wraps every field in double-quotes, so PostgreSQL \copy
        reads them correctly and the original text is preserved.
        """
        if x is None:
            return ''
        if isinstance(x, (list, dict, tuple, np.ndarray)):
            return str(x).replace('\x00', '')
        try:
            if pd.isna(x):
                return ''
        except (ValueError, TypeError):
            pass
        return str(x).replace('\x00', '')

    _NA_STRINGS = {'nan', 'NaN', 'NaT', 'None', '<NA>', 'NA'}

    # Drop 'Unnamed: *' columns (pandas index artifacts) from all tables
    # before generating SQL so they don't end up in the schema.
    for table_name, table in db.table_dict.items():
        unnamed_cols = [c for c in table.df.columns if c.startswith('Unnamed')]
        if unnamed_cols:
            print(f"  Dropping artifact columns from {table_name}: {unnamed_cols}")
            table.df.drop(columns=unnamed_cols, inplace=True)

    print("Exporting full CSV files (no sampling)...")
    for table_name, table in db.table_dict.items():
        csv_path = output_dir / f"{table_name}.csv"
        df_export = table.df.copy()

        # Convert every column to plain Python strings so pandas' C-level CSV
        # writer never encounters type-specific escaping issues.  Temp tables
        # are all-TEXT, so this is safe for the downstream SQL import.
        for col in df_export.columns:
            if df_export[col].dtype == 'object':
                df_export[col] = df_export[col].apply(clean_value)
            else:
                # Numeric / datetime / bool → str, then blank out NA tokens.
                # Use high precision for floats to avoid rounding during the
                # string round-trip through CSV.
                if df_export[col].dtype in ('float32', 'float64'):
                    df_export[col] = df_export[col].astype(str).where(
                        ~df_export[col].astype(str).isin(_NA_STRINGS), other=''
                    )
                else:
                    df_export[col] = df_export[col].astype(str).where(
                        ~df_export[col].astype(str).isin(_NA_STRINGS), other=''
                    )

        df_export.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"   {table_name}.csv ({len(table.df):,} rows)")
    print()

    print("Generating SQL DDL for full import...")
    sql_content = generate_complete_sql(db, dataset_name)

    dataset_short = dataset_name.replace('rel-', '')
    sql_file = output_dir / f"import_{dataset_short}_full.sql"
    sql_file.write_text(sql_content)

    print(f"SQL script generated: {sql_file}")
    print()

    print("Dataset Statistics:")
    print(f"   Tables: {len(db.table_dict)}")
    total_rows = 0
    for table_name, table in db.table_dict.items():
        pkey = get_primary_key_column(table)
        fkeys = len(table.fkey_col_to_pkey_table) if hasattr(table, 'fkey_col_to_pkey_table') else 0
        rows = len(table.df)
        total_rows += rows
        print(f"   - {table_name:30s} {rows:10,} rows, {len(table.df.columns):2} cols, PK: {pkey}, FKs: {fkeys}")
    print(f"\n   Total rows: {total_rows:,}")
    print()

    print("=" * 60)
    print("Generation complete")
    print("=" * 60)
    print()
    print("Output directory:", output_dir)
    print("SQL script:", sql_file)
    print()
    print("Next steps:")
    print(f"  ./import_relbench_full.sh {dataset_name}")


if __name__ == "__main__":
    main()
