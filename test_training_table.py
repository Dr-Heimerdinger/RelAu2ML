import os, sys
import numpy as np
import pandas as pd
from typing import Optional
import duckdb
from plexe.relbench.base import Database, Table, EntityTask, TaskType, Dataset
from plexe.relbench.metrics import accuracy, f1, roc_auc, average_precision

from workdir.rel_hm_user_churn_2.dataset import GenDataset
from workdir.rel_hm_user_churn_2.task import GenTask

from plexe.relbench.tasks.hm import UserChurnTask

csv_dir = '/home/ta/kl/plexe-clone/workdir/rel_hm_user_churn_2/csv_files'
dataset = GenDataset(csv_dir=csv_dir)
gen_task = GenTask(dataset)
root_task = UserChurnTask(dataset)

db = dataset.get_db()

print("=" * 60)
print("Training Table from GenTask:")
gen_table = gen_task.get_table("train")
print(gen_table)
print(f"\nShape: {gen_table.df.shape}")

print("\n" + "=" * 60)
print("Training Table from UserChurnTask (author):")
author_table = root_task.get_table("train")
print(author_table)
print(f"\nShape: {author_table.df.shape}")

# Compare
print("\n" + "=" * 60)
print("COMPARISON:")
print(f"  GenTask rows: {len(gen_table.df)}")
print(f"  UserChurnTask rows:    {len(author_table.df)}")
print(f"  Row counts match:      {len(gen_table.df) == len(author_table.df)}")

# Check if the dataframes have the same content (sort both for comparison)
gen_cols = list(gen_table.df.columns)
author_cols = list(author_table.df.columns)

# Assume the last column is the target, so we sort by all other columns (e.g. timestamp, driver_id)
gen_sort_cols = gen_cols[:-1] if len(gen_cols) > 1 else gen_cols
author_sort_cols = author_cols[:-1] if len(author_cols) > 1 else author_cols

gen_sorted = gen_table.df.sort_values(
    gen_sort_cols
).reset_index(drop=True)
author_sorted = author_table.df.sort_values(
    author_sort_cols
).reset_index(drop=True)

if len(gen_sorted) == len(author_sorted):
    if len(gen_cols) == len(author_cols):
        # Rename gen columns to match author columns to compare by position
        gen_mapped = gen_sorted.copy()
        gen_mapped.columns = author_cols
        
        exact_match = gen_mapped.equals(author_sorted)
        print(f"  Exact content match:   {exact_match}")    
        
        if not exact_match:
            # Check column by column
            for gen_c, auth_c in zip(gen_cols, author_cols):
                col_match = gen_mapped[auth_c].equals(author_sorted[auth_c])
                print(f"    Column '{gen_c}' vs '{auth_c}' match: {col_match}")
                if not col_match:
                    print(f"      Types: Gen='{gen_mapped[auth_c].dtype}' | Author='{author_sorted[auth_c].dtype}'")
                    try:
                        # Identify differing records and NaNs mismatch handling
                        diff_mask = (gen_mapped[auth_c] != author_sorted[auth_c]) & ~(gen_mapped[auth_c].isna() & author_sorted[auth_c].isna())
                        n_diff = diff_mask.sum()
                        print(f"    Differences: {n_diff}/{len(gen_sorted)}")
                        
                        if n_diff > 0:
                            print("    Sample differences (GenTask vs AuthorTask):")
                            # Get the indices where differences occur
                            diff_indices = gen_mapped.index[diff_mask]
                            
                            # Show up to 5 differences
                            for i, idx in enumerate(diff_indices[:5]):
                                print(f"      Row {idx}: Gen='{gen_mapped.loc[idx, auth_c]}' | Author='{author_sorted.loc[idx, auth_c]}'")
                            
                            if n_diff > 5:
                                print(f"      ... and {n_diff - 5} more differences.")
                                
                    except Exception as e:
                        print(f"    Could not count differences: {e}")
    else:
        print("  Cannot compare exact matches: differing number of columns.")
        print(f"  Gen columns: {gen_cols}")
        print(f"  Author columns: {author_cols}")