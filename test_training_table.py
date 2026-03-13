import os
import sys
import importlib
import inspect
import numpy as np
import pandas as pd
from typing import Optional
import duckdb
from plexe.relbench.base import Database, Table, EntityTask, TaskType, Dataset

# ─────────────────────────────────────────────
# Step 1: Nhập đường dẫn folder workdir
# ─────────────────────────────────────────────
print("=" * 60)
print("TEST TRAINING TABLE — Interactive Setup")
print("=" * 60)

workdir_root = "/data/anhtdt/RelAu2ML/workdir"

folder_name = input(
    "\n[1] Nhập tên folder trong workdir (ví dụ: rel_f1_driver_top3_4): "
).strip()

folder_path = os.path.join(workdir_root, folder_name)
if not os.path.isdir(folder_path):
    print(f"❌ Không tìm thấy thư mục: {folder_path}")
    sys.exit(1)

# Thêm workdir vào sys.path để import được module
sys.path.insert(0, os.path.dirname(workdir_root))

# Tự động xác định csv_files dir
csv_dir = os.path.join(folder_path, "csv_files")
if not os.path.isdir(csv_dir):
    # Tìm bất kỳ thư mục con chứa csv
    candidates = [
        os.path.join(folder_path, d)
        for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
        and d not in ("__pycache__",)
    ]
    if candidates:
        csv_dir = candidates[0]
        print(f"  ℹ️  Dùng csv_dir: {csv_dir}")
    else:
        print(f"❌ Không tìm thấy thư mục csv_files trong: {folder_path}")
        sys.exit(1)

# Import GenDataset và GenTask từ folder đã chọn
try:
    dataset_mod = importlib.import_module(f"workdir.{folder_name}.dataset")
    GenDataset = dataset_mod.GenDataset
except Exception as e:
    print(f"❌ Không thể import GenDataset từ workdir/{folder_name}/dataset.py: {e}")
    sys.exit(1)

try:
    task_mod = importlib.import_module(f"workdir.{folder_name}.task")
    GenTask = task_mod.GenTask
except Exception as e:
    print(f"❌ Không thể import GenTask từ workdir/{folder_name}/task.py: {e}")
    sys.exit(1)

print(f"  ✅ Đã tải GenDataset & GenTask từ workdir/{folder_name}")

# ─────────────────────────────────────────────
# Step 2: Nhập tên task (module trong plexe.relbench.tasks)
# ─────────────────────────────────────────────
available_tasks = ["f1", "amazon", "avito", "event", "hm", "stack", "trial"]
print(f"\n[2] Các task module khả dụng: {', '.join(available_tasks)}")
task_module_name = input("    Nhập tên task module (ví dụ: f1): ").strip()

try:
    author_task_mod = importlib.import_module(f"plexe.relbench.tasks.{task_module_name}")
except Exception as e:
    print(f"❌ Không thể import plexe.relbench.tasks.{task_module_name}: {e}")
    sys.exit(1)

# Lấy danh sách class Task có trong module vừa import
task_classes = {
    name: cls
    for name, cls in inspect.getmembers(author_task_mod, inspect.isclass)
    if cls.__module__ == author_task_mod.__name__
}

if not task_classes:
    print(f"❌ Không tìm thấy class Task nào trong module '{task_module_name}'")
    sys.exit(1)

# ─────────────────────────────────────────────
# Step 3: Chọn class Task của tác giả
# ─────────────────────────────────────────────
print(f"\n[3] Các class Task có trong module '{task_module_name}':")
for i, name in enumerate(task_classes.keys(), 1):
    print(f"    {i}. {name}")

task_class_names = list(task_classes.keys())
try:
    class_index = int(input("    Nhập số thứ tự class Task: ").strip())
    if not (1 <= class_index <= len(task_class_names)):
        raise ValueError
except ValueError:
    print(f"❌ Vui lòng nhập số từ 1 đến {len(task_class_names)}")
    sys.exit(1)

class_name = task_class_names[class_index - 1]
AuthorTaskClass = task_classes[class_name]
print(f"  ✅ Đã chọn: {class_name}")

# ─────────────────────────────────────────────
# Khởi tạo dataset và các task
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Đang khởi tạo dataset và task...")

dataset = GenDataset(csv_dir=csv_dir)
gen_task = GenTask(dataset)
root_task = AuthorTaskClass(dataset)

db = dataset.get_db()

# ─────────────────────────────────────────────
# Hiển thị: Training Table từ GenTask
# ─────────────────────────────────────────────
print("=" * 60)
print("Training Table from GenTask:")
gen_table = gen_task.get_table("train")
print(gen_table)
print(f"\nShape: {gen_table.df.shape}")

# ─────────────────────────────────────────────
# Hiển thị: Training Table từ Author Task
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Training Table from {class_name} (author):")
author_table = root_task.get_table("train")
print(author_table)
print(f"\nShape: {author_table.df.shape}")

# ─────────────────────────────────────────────
# So sánh
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON:")
print(f"  GenTask rows:      {len(gen_table.df)}")
print(f"  {class_name} rows: {len(author_table.df)}")
print(f"  Row counts match:  {len(gen_table.df) == len(author_table.df)}")

# Check if the dataframes have the same content (sort both for comparison)
gen_cols = list(gen_table.df.columns)
author_cols = list(author_table.df.columns)

# Assume the last column is the target, so we sort by all other columns
gen_sort_cols = gen_cols[:-1] if len(gen_cols) > 1 else gen_cols
author_sort_cols = author_cols[:-1] if len(author_cols) > 1 else author_cols

gen_sorted = gen_table.df.sort_values(gen_sort_cols).reset_index(drop=True)
author_sorted = author_table.df.sort_values(author_sort_cols).reset_index(drop=True)

if len(gen_cols) == len(author_cols):
    # Rename gen columns to match author columns to compare by content
    gen_mapped = gen_sorted.copy()
    gen_mapped.columns = author_cols

    # Calculate identical rows using merge (handles duplicates correctly for multiset intersection)
    # We use all columns for the intersection
    try:
        # subset=None means use all columns
        common_rows = pd.merge(gen_mapped, author_sorted, how='inner')
        # However, simple merge doesn't handle duplicate rows as a multiset intersection would if there are many of them
        # A more precise way for multiset intersection:
        gen_counts = gen_mapped.value_counts(dropna=False).reset_index(name='count_gen')
        auth_counts = author_sorted.value_counts(dropna=False).reset_index(name='count_auth')
        
        # Merge counts on all data columns
        merged_counts = pd.merge(gen_counts, auth_counts, on=author_cols, how='inner')
        matching_rows_count = merged_counts[['count_gen', 'count_auth']].min(axis=1).sum()
        
        percentage_gen = (matching_rows_count / len(gen_sorted)) * 100
        percentage_auth = (matching_rows_count / len(author_sorted)) * 100
        print(f"  Identical rows:        {matching_rows_count}")
        print(f"  % of GenTask:          {percentage_gen:.2f}%")
        print(f"  % of {class_name}:      {percentage_auth:.2f}%")
    except Exception as e:
        print(f"  Could not calculate identical rows: {e}")

    if len(gen_sorted) == len(author_sorted):
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
                        diff_mask = (
                            (gen_mapped[auth_c] != author_sorted[auth_c])
                            & ~(gen_mapped[auth_c].isna() & author_sorted[auth_c].isna())
                        )
                        n_diff = diff_mask.sum()
                        print(f"    Differences: {n_diff}/{len(gen_sorted)}")

                        if n_diff > 0:
                            print("    Sample differences (GenTask vs AuthorTask):")
                            diff_indices = gen_mapped.index[diff_mask]
                            for i, idx in enumerate(diff_indices[:5]):
                                print(
                                    f"      Row {idx}: Gen='{gen_mapped.loc[idx, auth_c]}' | Author='{author_sorted.loc[idx, auth_c]}'"
                                )
                            if n_diff > 5:
                                print(f"      ... and {n_diff - 5} more differences.")
                    except Exception as e:
                        print(f"    Could not count differences: {e}")
    else:
        print(f"\n  Note: Row counts differ ({len(gen_sorted)} vs {len(author_sorted)}).")
else:
    print("  Cannot compare exact matches: differing number of columns.")
    print(f"  Gen columns:    {gen_cols}")
    print(f"  Author columns: {author_cols}")