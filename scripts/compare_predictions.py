#!/usr/bin/env python3
"""
Re-run model inference from a workdir session and compare predictions against
author ground truth tables.

Usage:
    python scripts/compare_predictions.py \\
        --session session-118b7368-9b68-4be4-9765-cf62de1abe12 \\
        --dataset rel-f1 --task driver-dnf

    python scripts/compare_predictions.py \\
        --session session-118b7368-9b68-4be4-9765-cf62de1abe12 \\
        --dataset rel-f1 --task driver-dnf \\
        --output comparison.json
"""

import argparse
import copy
import importlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from plexe.relbench.base import Table
from plexe.relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from plexe.relbench.modeling.utils import get_stype_proposal
from scripts.task_registry import DATASET_TASK_MAP


# ---------------------------------------------------------------------------
# Model definition (must match train_script.py structure)
# ---------------------------------------------------------------------------
class Model(torch.nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels, out_channels):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr="sum",
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm="batch_norm",
            num_layers=1,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        return self.head(x_dict[entity_table][:seed_time.size(0)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_session_modules(session_path):
    """Dynamically import GenDataset and GenTask from a session directory."""
    # Add session dir to path for the import
    if session_path not in sys.path:
        sys.path.insert(0, session_path)

    folder_name = os.path.basename(session_path)
    workdir_parent = os.path.dirname(os.path.dirname(session_path))

    # Also add parent of workdir so `workdir.{folder}.module` pattern works
    if workdir_parent not in sys.path:
        sys.path.insert(0, workdir_parent)

    try:
        dataset_mod = importlib.import_module(f"workdir.{folder_name}.dataset")
        GenDataset = dataset_mod.GenDataset
    except Exception as e:
        raise RuntimeError(f"Cannot import GenDataset from {session_path}/dataset.py: {e}")

    try:
        task_mod = importlib.import_module(f"workdir.{folder_name}.task")
        GenTask = task_mod.GenTask
    except Exception as e:
        raise RuntimeError(f"Cannot import GenTask from {session_path}/task.py: {e}")

    return GenDataset, GenTask


def run_inference(model, loader, device, entity_table):
    """Run model inference and return numpy predictions."""
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
    if not pred_list:
        return np.array([])
    return torch.cat(pred_list, dim=0).numpy()


def format_table(split, gen_metrics, author_metrics, gen_rows, author_rows, matched_rows):
    """Format a comparison table for display."""
    # Collect all metric names
    all_metrics = sorted(set(list(gen_metrics.keys()) + list(author_metrics.keys())))

    lines = [f"\nSplit: {split}"]
    col1_w, col2_w, col3_w = 22, 14, 14
    sep = f"+{'-' * col1_w}+{'-' * col2_w}+{'-' * col3_w}+"
    hdr = f"|{'Metric':<{col1_w}}|{'GenTask':^{col2_w}}|{'Author Task':^{col3_w}}|"

    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)

    for m in all_metrics:
        gv = f"{gen_metrics[m]:.4f}" if m in gen_metrics else "N/A"
        av = f"{author_metrics[m]:.4f}" if m in author_metrics else "N/A"
        lines.append(f"|{m:<{col1_w}}|{gv:^{col2_w}}|{av:^{col3_w}}|")

    lines.append(sep)
    lines.append(f"|{'Table rows':<{col1_w}}|{str(gen_rows):^{col2_w}}|{str(author_rows):^{col3_w}}|")
    lines.append(f"|{'Matched rows':<{col1_w}}|{str(matched_rows):^{col2_w}}|{'':^{col3_w}}|")
    lines.append(sep)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions against author ground truth tables"
    )
    parser.add_argument("--session", required=True,
                        help="Session folder name or full path")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. rel-f1)")
    parser.add_argument("--task", required=True,
                        help="Task name (e.g. driver-dnf)")
    parser.add_argument("--workdir", type=str, default="./workdir",
                        help="Root workdir path (default: ./workdir)")
    parser.add_argument("--author-tables-dir", type=str, default="./data/author_tables",
                        help="Path to author tables (default: ./data/author_tables)")
    parser.add_argument("--splits", type=str, default="val,test",
                        help="Splits to compare (default: val,test)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save comparison JSON to file (default: print to stdout)")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]

    # Validate dataset/task in registry
    if args.dataset not in DATASET_TASK_MAP:
        print(f"Error: Unknown dataset '{args.dataset}'. "
              f"Available: {list(DATASET_TASK_MAP.keys())}")
        sys.exit(1)

    ds_info = DATASET_TASK_MAP[args.dataset]
    if args.task not in ds_info["tasks"]:
        print(f"Error: Unknown task '{args.task}' for dataset '{args.dataset}'. "
              f"Available: {list(ds_info['tasks'].keys())}")
        sys.exit(1)

    AuthorDatasetClass = ds_info["DatasetClass"]
    AuthorTaskClass = ds_info["tasks"][args.task]

    # Resolve session path
    if os.path.isabs(args.session) and os.path.isdir(args.session):
        session_path = args.session
    else:
        session_path = os.path.join(args.workdir, args.session)
    session_path = os.path.abspath(session_path)

    if not os.path.isdir(session_path):
        print(f"Error: Session directory not found: {session_path}")
        sys.exit(1)

    # Check for required artifacts
    model_path = os.path.join(session_path, "best_model.pt")
    results_path = os.path.join(session_path, "training_results.json")
    csv_dir = os.path.join(session_path, "csv_files")

    for path, label in [(model_path, "best_model.pt"), (csv_dir, "csv_files/")]:
        if not os.path.exists(path):
            print(f"Error: Required artifact not found: {path}")
            sys.exit(1)

    # Load training results if available
    training_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            training_results = json.load(f)
        print(f"Loaded training_results.json (existing metrics: "
              f"val={training_results.get('val_metrics', {})}, "
              f"test={training_results.get('test_metrics', {})})")

    # -----------------------------------------------------------------------
    # Step 1: Load session modules and build GenTask pipeline
    # -----------------------------------------------------------------------
    print("\n[1/4] Loading session modules...")
    GenDataset, GenTask = load_session_modules(session_path)

    dataset = GenDataset(csv_dir=csv_dir)
    gen_task = GenTask(dataset)
    db = dataset.get_db()
    entity_table = gen_task.entity_table

    # Get tables for requested splits
    gen_tables = {}
    for split in splits:
        gen_tables[split] = gen_task.get_table(split)

    # -----------------------------------------------------------------------
    # Step 2: Build graph and model, run inference
    # -----------------------------------------------------------------------
    print("[2/4] Building graph and loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Text embedder (optional, same as train_script)
    text_embedder_cfg = None
    try:
        from sentence_transformers import SentenceTransformer
        from torch import Tensor
        from torch_frame.config.text_embedder import TextEmbedderConfig
        from typing import List, Optional as Opt

        class GloveTextEmbedding:
            def __init__(self, device=None):
                self.model = SentenceTransformer(
                    "sentence-transformers/average_word_embeddings_glove.6B.300d",
                    device=device,
                )
            def __call__(self, sentences: list) -> Tensor:
                return torch.from_numpy(self.model.encode(sentences))

        text_embedder_cfg = TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        )
        print("  Using GloVe text embeddings")
    except Exception:
        print("  Text embeddings not available, using None")

    col_to_stype_dict = get_stype_proposal(db)
    cache_dir = os.path.join(session_path, "cache")
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=cache_dir,
    )

    # Build loaders
    loader_dict = {}
    for split in splits:
        table_input = get_node_train_table_input(table=gen_tables[split], task=gen_task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[32] * 2,
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=128,
            temporal_strategy="uniform",
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )

    # Build and load model
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=2,
        channels=128,
        out_channels=1,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  Model loaded from {model_path}")

    # Run inference
    print("[3/4] Running inference...")
    gen_predictions = {}
    gen_metrics = {}
    for split in splits:
        pred = run_inference(model, loader_dict[split], device, entity_table)
        gen_predictions[split] = pred
        if len(pred) > 0:
            gen_metrics[split] = gen_task.evaluate(pred, gen_tables[split])
            print(f"  [{split}] GenTask metrics: {gen_metrics[split]} ({len(pred)} rows)")
        else:
            gen_metrics[split] = {}
            print(f"  [{split}] No predictions generated")

    # -----------------------------------------------------------------------
    # Step 3: Load author ground truth and compare
    # -----------------------------------------------------------------------
    print("[4/4] Comparing with author tables...")

    author_dataset = AuthorDatasetClass()
    author_task = AuthorTaskClass(author_dataset)

    comparison_results = {}

    for split in splits:
        author_table_path = os.path.join(
            args.author_tables_dir, args.dataset, args.task, f"{split}.parquet"
        )

        if not os.path.exists(author_table_path):
            print(f"  [{split}] Author table not found: {author_table_path}")
            print(f"         Run: python scripts/download_author_tables.py "
                  f"--dataset {args.dataset} --task {args.task}")
            comparison_results[split] = {"error": "Author table not found"}
            continue

        author_table = Table.load(author_table_path)
        gen_table = gen_tables[split]
        pred = gen_predictions[split]

        if len(pred) == 0:
            comparison_results[split] = {"error": "No predictions available"}
            continue

        # Row alignment: merge on (entity_col, time_col)
        gen_df = gen_table.df.copy()
        gen_df["__pred__"] = pred

        author_df = author_table.df.copy()

        # Identify join columns (entity_col and time_col from gen_task)
        gen_entity_col = gen_task.entity_col
        gen_time_col = gen_task.time_col
        author_entity_col = author_task.entity_col
        author_time_col = author_task.time_col

        # Merge gen predictions with author table on entity + time
        merged = pd.merge(
            gen_df[[gen_entity_col, gen_time_col, "__pred__"]].rename(
                columns={gen_entity_col: "entity", gen_time_col: "time"}
            ),
            author_df.rename(
                columns={author_entity_col: "entity", author_time_col: "time"}
            ),
            on=["entity", "time"],
            how="inner",
        )

        matched_rows = len(merged)
        gen_rows = len(gen_df)
        author_rows = len(author_df)

        author_metrics_split = {}
        if matched_rows > 0:
            # Build a matched author table for evaluation
            matched_author_df = merged.rename(
                columns={"entity": author_entity_col, "time": author_time_col}
            )
            # Drop the __pred__ col from matched_author_df for Table construction
            author_table_cols = list(author_table.df.columns)
            matched_author_table = Table(
                df=matched_author_df[author_table_cols],
                fkey_col_to_pkey_table=author_table.fkey_col_to_pkey_table,
                pkey_col=author_table.pkey_col,
                time_col=author_table.time_col,
            )
            matched_pred = merged["__pred__"].to_numpy()

            try:
                author_metrics_split = author_task.evaluate(
                    matched_pred, matched_author_table
                )
            except Exception as e:
                author_metrics_split = {"error": str(e)}
                print(f"  [{split}] Author evaluation error: {e}")

        # Display comparison
        display = format_table(
            split,
            gen_metrics.get(split, {}),
            author_metrics_split if not isinstance(author_metrics_split, dict) or "error" not in author_metrics_split else {},
            gen_rows,
            author_rows,
            matched_rows,
        )
        print(display)

        comparison_results[split] = {
            "gen_metrics": {k: float(v) for k, v in gen_metrics.get(split, {}).items()},
            "author_metrics": (
                {k: float(v) for k, v in author_metrics_split.items()}
                if isinstance(author_metrics_split, dict) and "error" not in author_metrics_split
                else author_metrics_split
            ),
            "gen_rows": gen_rows,
            "author_rows": author_rows,
            "matched_rows": matched_rows,
        }

    # Output JSON
    output_data = {
        "session": args.session,
        "dataset": args.dataset,
        "task": args.task,
        "splits": comparison_results,
        "training_results": training_results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nComparison saved to {args.output}")
    else:
        print(f"\n--- JSON Output ---")
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
