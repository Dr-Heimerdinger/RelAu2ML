#!/usr/bin/env python3
"""
Download relbench author-defined task tables (val + test splits) and store as parquet.

Usage:
    python scripts/download_author_tables.py --dataset rel-f1
    python scripts/download_author_tables.py --dataset rel-f1 --task driver-dnf
    python scripts/download_author_tables.py  # all datasets, all tasks
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.task_registry import get_dataset_task_pairs
from typing import Any


def main():
    parser = argparse.ArgumentParser(
        description="Download author-defined relbench task tables as parquet"
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter to one dataset (e.g. rel-f1). Default: all")
    parser.add_argument("--task", type=str, default=None,
                        help="Filter to one task (e.g. driver-dnf). Default: all")
    parser.add_argument("--output-dir", type=str, default="./data/author_tables",
                        help="Root output directory (default: ./data/author_tables)")
    parser.add_argument("--splits", type=str, default="val,test",
                        help="Comma-separated splits to download (default: val,test)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if parquet/csv already exists")
    parser.add_argument("--no-csv", action="store_true",
                        help="Skip saving CSV files (only save parquet)")
    parser.add_argument(
        "--no-prepared-download",
        action="store_true",
        help=(
            "Do not use RelBench prepared dataset downloads. "
            "If set, datasets will be instantiated from raw sources via local DatasetClass(). "
            "Default: use prepared downloads when available."
        ),
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]

    try:
        pairs = list(get_dataset_task_pairs(args.dataset, args.task))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not pairs:
        print("No dataset-task pairs matched the given filters.")
        sys.exit(1)

    print(f"Will download {len(pairs)} task(s), splits: {splits}")
    print(f"Output dir: {args.output_dir}\n")

    # Cache instantiated datasets to avoid re-downloading per task
    dataset_cache = {}
    success_count = 0
    skip_count = 0
    fail_count = 0

    def _get_dataset_prepared(ds_name: str) -> Any:
        # Use the upstream `relbench` package's prepared dataset cache/download.
        # This matches the behavior of scripts like generate_relbench_sql_full.py.
        from relbench.datasets import get_dataset as rb_get_dataset

        return rb_get_dataset(ds_name, download=True)

    for ds_name, task_name, DatasetClass, TaskClass in pairs:
        print(f"--- {ds_name} / {task_name} ---")

        for split in splits:
            out_path = os.path.join(args.output_dir, ds_name, task_name, f"{split}.parquet")

            csv_path = os.path.splitext(out_path)[0] + ".csv"

            parquet_exists = os.path.exists(out_path)
            csv_exists = os.path.exists(csv_path)
            both_exist = parquet_exists and (args.no_csv or csv_exists)

            if both_exist and not args.force:
                print(f"  [{split}] Already exists, skipping (use --force to overwrite)")
                skip_count += 1
                continue

            try:
                # Instantiate dataset (cached)
                if ds_name not in dataset_cache:
                    if not args.no_prepared_download and ds_name.startswith("rel-"):
                        print("  Loading prepared dataset via relbench.get_dataset(download=True)...")
                        dataset_cache[ds_name] = _get_dataset_prepared(ds_name)
                    else:
                        print(f"  Instantiating {DatasetClass.__name__}...")
                        dataset_cache[ds_name] = DatasetClass()

                dataset = dataset_cache[ds_name]
                task = TaskClass(dataset)

                print(f"  [{split}] Fetching table...")
                table = task.get_table(split, mask_input_cols=False)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                if not parquet_exists or args.force:
                    table.save(out_path)
                    print(f"  [{split}] Saved parquet to {out_path} ({len(table)} rows)")

                if not args.no_csv and (not csv_exists or args.force):
                    table.df.to_csv(csv_path, index=False)
                    print(f"  [{split}] Saved CSV to {csv_path} ({len(table.df)} rows)")

                success_count += 1

            except Exception as e:
                print(f"  [{split}] FAILED: {e}")
                fail_count += 1

    print(f"\nDone! Saved: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
