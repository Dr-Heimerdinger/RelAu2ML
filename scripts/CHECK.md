# Author Table Download & Prediction Comparison

Two scripts for evaluating auto-generated GNN models against the official RelBench author-defined task tables.

## Prerequisites

Run all commands from the **project root** directory. These scripts are designed to run in the Docker environment where all dependencies (torch, torch_geometric, sklearn, etc.) are installed.

---

## Script 1: Download Author Tables

Downloads author-defined val/test ground truth tables and saves them as **parquet and CSV** files.

### Basic Usage

```bash
# Download all tasks for a single dataset
python scripts/download_author_tables.py --dataset rel-f1

# Download a specific task
python scripts/download_author_tables.py --dataset rel-f1 --task driver-dnf

# Download all datasets (takes a long time)
python scripts/download_author_tables.py

# Re-download even if files already exist
python scripts/download_author_tables.py --dataset rel-f1 --force

# Download only val split
python scripts/download_author_tables.py --dataset rel-f1 --splits val

# Custom output directory
python scripts/download_author_tables.py --dataset rel-f1 --output-dir /path/to/output
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | all | Dataset name (e.g. `rel-f1`, `rel-amazon`) |
| `--task` | all | Task name (e.g. `driver-dnf`, `user-churn`) |
| `--output-dir` | `./data/author_tables` | Root output directory |
| `--splits` | `val,test` | Comma-separated splits to download |
| `--force` | off | Re-download even if parquet/csv exists |
| `--no-csv` | off | Skip saving CSV files (only save parquet) |

### Output Structure

```
data/author_tables/
  rel-f1/
    driver-position/
      val.parquet
      val.csv
      test.parquet
      test.csv
    driver-dnf/
      val.parquet
      val.csv
      test.parquet
      test.csv
    driver-top3/
      val.parquet
      val.csv
      test.parquet
      test.csv
    driver-race-compete/
      val.parquet
      val.csv
      test.parquet
      test.csv
  rel-amazon/
    user-churn/
      val.parquet
      val.csv
      test.parquet
      test.csv
    ...
```

---

## Script 2: Compare Predictions

Re-runs model inference from a trained session and compares predictions against author ground truth.

### Prerequisites

1. A completed training session in `workdir/` (must contain `best_model.pt`, `csv_files/`, `dataset.py`, `task.py`)
2. Author tables already downloaded via Script 1

### Basic Usage

```bash
# Compare a session against author tables
python scripts/compare_predictions.py \
    --session session-118b7368-9b68-4be4-9765-cf62de1abe12 \
    --dataset rel-f1 \
    --task driver-dnf

# Save results to a JSON file
python scripts/compare_predictions.py \
    --session session-118b7368-9b68-4be4-9765-cf62de1abe12 \
    --dataset rel-f1 \
    --task driver-dnf \
    --output comparison.json

# Compare only the val split
python scripts/compare_predictions.py \
    --session session-118b7368-9b68-4be4-9765-cf62de1abe12 \
    --dataset rel-f1 \
    --task driver-dnf \
    --splits val

# Use a custom workdir or author tables path
python scripts/compare_predictions.py \
    --session session-118b7368-9b68-4be4-9765-cf62de1abe12 \
    --dataset rel-f1 \
    --task driver-dnf \
    --workdir /path/to/workdir \
    --author-tables-dir /path/to/author_tables

# Pass a full absolute session path instead of a folder name
python scripts/compare_predictions.py \
    --session /absolute/path/to/session-folder \
    --dataset rel-f1 \
    --task driver-dnf
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--session` | **(required)** | Session folder name or full path |
| `--dataset` | **(required)** | Dataset name (e.g. `rel-f1`) |
| `--task` | **(required)** | Task name (e.g. `driver-dnf`) |
| `--workdir` | `./workdir` | Root workdir path |
| `--author-tables-dir` | `./data/author_tables` | Path to downloaded author tables |
| `--splits` | `val,test` | Comma-separated splits to compare |
| `--output` | stdout | Save comparison JSON to file |

### Output

The script prints a side-by-side comparison table:

```
Split: val
+----------------------+--------------+--------------+
|Metric                |   GenTask    | Author Task  |
+----------------------+--------------+--------------+
|average_precision     |    0.8523    |    0.8312    |
|roc_auc               |    0.8234    |    0.8101    |
+----------------------+--------------+--------------+
|Table rows            |     1523     |     1489     |
|Matched rows          |     1456     |              |
+----------------------+--------------+--------------+
```

When `--output` is provided, results are also saved as JSON with this structure:

```json
{
  "session": "session-118b7368-...",
  "dataset": "rel-f1",
  "task": "driver-dnf",
  "splits": {
    "val": {
      "gen_metrics": {"roc_auc": 0.8234, "average_precision": 0.8523},
      "author_metrics": {"roc_auc": 0.8101, "average_precision": 0.8312},
      "gen_rows": 1523,
      "author_rows": 1489,
      "matched_rows": 1456
    },
    "test": { ... }
  },
  "training_results": { ... }
}
```

### What the Script Does (Step by Step)

1. **Loads session artifacts** -- Dynamically imports `GenDataset` and `GenTask` from the session's `dataset.py` and `task.py`
2. **Rebuilds the GNN pipeline** -- Constructs the heterogeneous graph, data loaders, and `Model` architecture, then loads `best_model.pt` weights
3. **Runs inference** -- Generates predictions on the requested splits and evaluates them against GenTask tables
4. **Compares with author tables** -- Loads author parquet files, aligns rows by `(entity_col, time_col)`, and evaluates predictions against author ground truth using the author task's metrics

---

## Available Datasets and Tasks

| Dataset | Tasks |
|---|---|
| `rel-f1` | `driver-position`, `driver-dnf`, `driver-top3`, `driver-race-compete` |
| `rel-amazon` | `user-churn`, `user-ltv`, `item-churn`, `item-ltv`, `user-item-purchase`, `user-item-rate`, `user-item-review` |
| `rel-hm` | `user-item-purchase`, `user-churn`, `item-sales` |
| `rel-stack` | `user-engagement`, `post-votes`, `user-badge`, `user-post-comment`, `post-post-related` |
| `rel-trial` | `study-outcome`, `study-adverse`, `site-success`, `condition-sponsor-run`, `site-sponsor-run` |
| `rel-event` | `user-attendance`, `user-repeat`, `user-ignore` |
| `rel-avito` | `ad-ctr`, `user-visits`, `user-clicks`, `user-ad-visit` |

---

## Typical Workflow

```bash
# 1. Download author tables for the dataset you care about
python scripts/download_author_tables.py --dataset rel-f1

# 2. Verify the parquet files were created
ls data/author_tables/rel-f1/driver-dnf/

# 3. Run comparison against a trained session
python scripts/compare_predictions.py \
    --session session-118b7368-9b68-4be4-9765-cf62de1abe12 \
    --dataset rel-f1 \
    --task driver-dnf \
    --output results/f1_driver_dnf_comparison.json
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Author table not found` | Run `download_author_tables.py` first for that dataset/task |
| `Required artifact not found: best_model.pt` | The session training did not complete successfully |
| `Cannot import GenDataset` | The session's `dataset.py` may have path issues -- check `csv_dir` inside it |
| `No predictions generated` | The data loader returned empty batches -- check that `csv_files/` has valid data |
| `Matched rows = 0` | The GenTask and author task may define different entity/time columns or time ranges |
| `ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'` | Install PyG C++ extensions: `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html` (pyg_lib may need the nightly index) |
