#!/usr/bin/env python3
"""
Re-run model inference from a workdir session and compare predictions against
author ground truth tables.  Includes an integrated fairness check that
verifies timestamp alignment, task-definition alignment, hyperparameter
consistency, and matched-row coverage before showing the metric comparison.

Usage:
    python scripts/compare_predictions.py \\
        --session session-4b1b925a-0e16-480b-88dd-98ae085708d9 \\
        --dataset rel-hm --task user-churn

    python scripts/compare_predictions.py \\
        --session session-4b1b925a-0e16-480b-88dd-98ae085708d9 \\
        --dataset rel-hm --task user-churn \\
        --output results/hm_churn.json

    # Skip the fairness banner (e.g. in CI)
    python scripts/compare_predictions.py ... --skip-fairness-check

    # Just print the hyper-param config that will be used
    python scripts/compare_predictions.py ... --show-config
"""

import argparse
import copy
import importlib
import json
import math
import os
import re
import sys
import types

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from plexe.relbench.base import Table, TaskType
from plexe.relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from plexe.relbench.modeling.utils import get_stype_proposal
from scripts.task_registry import DATASET_TASK_MAP

# ── ANSI colours (gracefully degraded when piped) ─────────────────────────────
_USE_COLOUR = sys.stdout.isatty()
GREEN  = "\033[92m" if _USE_COLOUR else ""
YELLOW = "\033[93m" if _USE_COLOUR else ""
RED    = "\033[91m" if _USE_COLOUR else ""
BOLD   = "\033[1m"  if _USE_COLOUR else ""
RESET  = "\033[0m"  if _USE_COLOUR else ""

def _ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def _warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def _fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def _info(msg): print(f"     {msg}")
def _hdr(msg):  print(f"\n{BOLD}{'─'*58}\n{msg}\n{'─'*58}{RESET}")


# ── Model (must mirror train_script.py exactly) ────────────────────────────────
class Model(torch.nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels, out_channels):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                nt: data[nt].tf.col_names_dict for nt in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr="sum",
            num_layers=num_layers,
        )
        self.head = MLP(channels, out_channels=out_channels,
                        norm="batch_norm", num_layers=1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for nt, rt in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rt
        x_dict = self.gnn(x_dict, batch.edge_index_dict,
                          batch.num_sampled_nodes_dict,
                          batch.num_sampled_edges_dict)
        return self.head(x_dict[entity_table][:seed_time.size(0)])


# ── Hyper-param config reader ──────────────────────────────────────────────────
# Author-recommended defaults (from snap-stanford/relbench/examples/gnn_entity.py)
_AUTHOR_DEFAULTS = {
    "num_layers":        2,
    "channels":          128,
    "num_neighbors":     128,   # author default; train_script often uses 128
    "batch_size":        512,   # author default; train_script often uses 512
    "temporal_strategy": "uniform",
    "tune_metric":       "roc_auc",
}

# Regex patterns for parsing train_script.py
_RE_NUM_NEIGHBORS = re.compile(r"num_neighbors\s*=\s*\[(\d+)\]\s*\*\s*(\d+)")
_RE_BATCH_SIZE    = re.compile(r"batch_size\s*=\s*(\d+)")
_RE_NUM_LAYERS    = re.compile(r"num_layers\s*=\s*(\d+)")
_RE_CHANNELS      = re.compile(r"\bchannels\s*=\s*(\d+)")
_RE_TUNE_METRIC   = re.compile(r'tune_metric\s*=\s*["\'](\w+)["\']')


def load_model_config(session_path: str, gen_task=None) -> dict:
    """
    Read hyper-parameters in priority order:
      1. session/model_config.json  (explicit — future-proof)
      2. session/train_script.py    (parse with regex)
      3. Author defaults from gnn_entity.py

    Also derives ``out_channels`` from the GenTask task_type when provided.
    """
    cfg = dict(_AUTHOR_DEFAULTS)  # start from author defaults
    source = "author defaults"

    # ── 1. model_config.json ──────────────────────────────────────────────
    config_path = os.path.join(session_path, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        cfg.update({k: v for k, v in saved.items() if k in cfg})
        source = "model_config.json"

    # ── 2. train_script.py regex fallback ────────────────────────────────
    else:
        ts_path = os.path.join(session_path, "train_script.py")
        if os.path.exists(ts_path):
            text = open(ts_path).read()
            m = _RE_NUM_NEIGHBORS.search(text)
            if m:
                cfg["num_neighbors"] = int(m.group(1))
                cfg["num_layers"] = int(m.group(2))   # num_layers = list length
            m = _RE_BATCH_SIZE.search(text)
            if m:
                cfg["batch_size"] = int(m.group(1))
            m = _RE_NUM_LAYERS.search(text)           # explicit num_layers= line
            if m:
                cfg["num_layers"] = int(m.group(1))
            m = _RE_CHANNELS.search(text)
            if m:
                cfg["channels"] = int(m.group(1))
            m = _RE_TUNE_METRIC.search(text)
            if m:
                cfg["tune_metric"] = m.group(1)
            source = "train_script.py (parsed)"

    # ── 3. Derive out_channels from task_type ────────────────────────────
    if gen_task is not None:
        tt = gen_task.task_type
        if tt == TaskType.MULTICLASS_CLASSIFICATION:
            cfg["out_channels"] = getattr(gen_task, "num_classes", 1)
        elif tt == TaskType.MULTILABEL_CLASSIFICATION:
            cfg["out_channels"] = getattr(gen_task, "num_labels", 1)
        else:
            cfg["out_channels"] = 1          # binary classification or regression
    else:
        cfg.setdefault("out_channels", 1)

    cfg["_source"] = source
    return cfg


# ── Session module loader ──────────────────────────────────────────────────────
def load_session_modules(session_path):
    """Dynamically import GenDataset and GenTask from a session directory."""
    if session_path not in sys.path:
        sys.path.insert(0, session_path)
    folder_name   = os.path.basename(session_path)
    workdir_parent = os.path.dirname(os.path.dirname(session_path))
    if workdir_parent not in sys.path:
        sys.path.insert(0, workdir_parent)
    try:
        dm = importlib.import_module(f"workdir.{folder_name}.dataset")
        GenDataset = dm.GenDataset
    except Exception as e:
        raise RuntimeError(f"Cannot import GenDataset from {session_path}/dataset.py: {e}")
    try:
        tm = importlib.import_module(f"workdir.{folder_name}.task")
        GenTask = tm.GenTask
    except Exception as e:
        raise RuntimeError(f"Cannot import GenTask from {session_path}/task.py: {e}")
    return GenDataset, GenTask


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader, device, entity_table):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            preds.append(pred.detach().cpu())
    if not preds:
        return np.array([])
    return torch.cat(preds, dim=0).numpy()


# ── Fairness check ─────────────────────────────────────────────────────────────
def run_fairness_check(gen_dataset, gen_task,
                       author_dataset, author_task,
                       cfg: dict,
                       author_tables_dir: str,
                       dataset_name: str, task_name: str,
                       splits: list) -> dict:
    """
    Print a compact fairness report and return a summary dict with keys:
      timestamp_ok, task_ok, coverage (per split).
    """
    _hdr("Fairness Check")
    summary = {"timestamp_ok": True, "task_ok": True, "coverage": {}}

    # ── 1. Timestamp alignment ────────────────────────────────────────────
    print(f"\n  {'':30s} {'GenDataset':>20}  {'AuthorDataset':>20}")
    for attr in ("val_timestamp", "test_timestamp"):
        gv = getattr(gen_dataset, attr, None)
        av = getattr(author_dataset, attr, None)
        diff = abs((gv - av).days) if gv is not None and av is not None else None
        tag = ""
        if diff is not None:
            if diff == 0:   tag = f"{GREEN}✅ same{RESET}"
            elif diff <= 7: tag = f"{YELLOW}⚠️  {diff}d diff{RESET}"
            else:           tag = f"{RED}❌ {diff}d diff{RESET}"; summary["timestamp_ok"] = False
        print(f"  {attr:30s} {str(gv):>20}  {str(av):>20}  {tag}")

    if not summary["timestamp_ok"]:
        print(f"\n  {RED}→ Timestamp mismatch means train/test splits are DIFFERENT.")
        print(f"    matched_rows will likely be ~0 and author_metrics unreliable.{RESET}")

    # ── 2. Task-definition alignment ──────────────────────────────────────
    print()
    task_fields = ["entity_col", "entity_table", "time_col", "target_col",
                   "timedelta", "task_type"]
    for field in task_fields:
        gv = str(getattr(gen_task,    field, "N/A"))
        av = str(getattr(author_task, field, "N/A"))
        if gv == av:
            _ok(f"{field:<20} = {gv}")
        else:
            _fail(f"{field:<20}: GenTask={gv!r}  AuthorTask={av!r}")
            summary["task_ok"] = False

    if not summary["task_ok"]:
        print(f"\n  {RED}→ Task-definition mismatch: metric columns measure DIFFERENT things.{RESET}")

    # ── 3. Hyperparameter transparency ───────────────────────────────────
    print(f"\n  Hyperparams (source: {BOLD}{cfg['_source']}{RESET})")
    _info(f"num_layers={cfg['num_layers']}  channels={cfg['channels']}  "
          f"out_channels={cfg['out_channels']}")
    _info(f"num_neighbors={cfg['num_neighbors']}  batch_size={cfg['batch_size']}  "
          f"temporal_strategy={cfg['temporal_strategy']}")

    ref_nn = 128
    if cfg["num_neighbors"] != ref_nn:
        _warn(f"num_neighbors={cfg['num_neighbors']} differs from author ref ({ref_nn})")
    else:
        _ok(f"num_neighbors matches author reference ({ref_nn})")

    # ── 4. matched_rows coverage ──────────────────────────────────────────
    print()
    for split in splits:
        parquet = os.path.join(author_tables_dir, dataset_name, task_name, f"{split}.parquet")
        if not os.path.exists(parquet):
            _warn(f"[{split}] Author parquet not found — run download_author_tables.py first")
            summary["coverage"][split] = None
            continue

        author_tbl = Table.load(parquet)
        try:
            gen_tbl = gen_task.get_table(split, mask_input_cols=False)
        except Exception as e:
            _warn(f"[{split}] Could not build GenTask table: {e}")
            summary["coverage"][split] = None
            continue

        merged = pd.merge(
            gen_tbl.df[[gen_task.entity_col, gen_task.time_col]].rename(
                columns={gen_task.entity_col: "_e", gen_task.time_col: "_t"}),
            author_tbl.df[[author_task.entity_col, author_task.time_col]].rename(
                columns={author_task.entity_col: "_e", author_task.time_col: "_t"}),
            on=["_e", "_t"], how="inner",
        )
        gen_n    = len(gen_tbl.df)
        auth_n   = len(author_tbl.df)
        match_n  = len(merged)
        cov      = match_n / auth_n if auth_n else 0
        summary["coverage"][split] = cov

        cov_str = f"{cov:.1%}  ({match_n:,} / {auth_n:,} author rows)"
        if cov >= 0.90:   _ok(f"[{split}] Coverage {cov_str}")
        elif cov >= 0.70: _warn(f"[{split}] Coverage {cov_str} — some noise in author_metrics")
        else:             _fail(f"[{split}] Coverage {cov_str} — author_metrics UNRELIABLE")

    return summary


# ── Comparison table printer ───────────────────────────────────────────────────
def format_table(split, gen_metrics, author_metrics, gen_rows, author_rows, matched_rows):
    all_m = sorted(set(list(gen_metrics) + list(author_metrics)))
    w1, w2, w3 = 22, 14, 14
    sep = f"+{'-'*w1}+{'-'*w2}+{'-'*w3}+"
    lines = [f"\nSplit: {split}", sep,
             f"|{'Metric':<{w1}}|{'GenTask':^{w2}}|{'Author Task':^{w3}}|", sep]
    for m in all_m:
        gv = f"{gen_metrics[m]:.4f}"    if m in gen_metrics    else "N/A"
        av = f"{author_metrics[m]:.4f}" if m in author_metrics else "N/A"
        lines.append(f"|{m:<{w1}}|{gv:^{w2}}|{av:^{w3}}|")
    lines += [sep,
              f"|{'Table rows':<{w1}}|{str(gen_rows):^{w2}}|{str(author_rows):^{w3}}|",
              f"|{'Matched rows':<{w1}}|{str(matched_rows):^{w2}}|{'':^{w3}}|",
              sep]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions against author ground truth (with fairness check)"
    )
    parser.add_argument("--session",  required=True,
                        help="Session folder name or full path")
    parser.add_argument("--dataset",  required=True,
                        help="Dataset name (e.g. rel-hm)")
    parser.add_argument("--task",     required=True,
                        help="Task name (e.g. user-churn)")
    parser.add_argument("--workdir",  default="./workdir",
                        help="Root workdir path (default: ./workdir)")
    parser.add_argument("--author-tables-dir", default="./data/author_tables",
                        help="Path to downloaded author tables")
    parser.add_argument("--splits",   default="val,test",
                        help="Comma-separated splits (default: val,test)")
    parser.add_argument("--output",   default=None,
                        help="Save comparison JSON to file")
    parser.add_argument("--skip-fairness-check", action="store_true",
                        help="Skip the fairness-check section")
    parser.add_argument("--show-config", action="store_true",
                        help="Print the resolved model config and exit")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]

    # ── Validate dataset / task ───────────────────────────────────────────
    if args.dataset not in DATASET_TASK_MAP:
        print(f"Error: Unknown dataset '{args.dataset}'. "
              f"Available: {list(DATASET_TASK_MAP.keys())}")
        sys.exit(1)
    ds_info = DATASET_TASK_MAP[args.dataset]
    if args.task not in ds_info["tasks"]:
        print(f"Error: Unknown task '{args.task}' for '{args.dataset}'. "
              f"Available: {list(ds_info['tasks'].keys())}")
        sys.exit(1)

    AuthorDatasetClass = ds_info["DatasetClass"]
    AuthorTaskClass    = ds_info["tasks"][args.task]

    # ── Resolve session path ─────────────────────────────────────────────
    if os.path.isabs(args.session) and os.path.isdir(args.session):
        session_path = args.session
    else:
        session_path = os.path.abspath(os.path.join(args.workdir, args.session))

    if not os.path.isdir(session_path):
        print(f"Error: Session directory not found: {session_path}")
        sys.exit(1)

    model_path   = os.path.join(session_path, "best_model.pt")
    results_path = os.path.join(session_path, "training_results.json")
    csv_dir      = os.path.join(session_path, "csv_files")

    for path, label in [(model_path, "best_model.pt"), (csv_dir, "csv_files/")]:
        if not os.path.exists(path):
            print(f"Error: Required artifact not found: {path}")
            sys.exit(1)

    # ── Load training results ────────────────────────────────────────────
    training_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            training_results = json.load(f)
        print(f"Loaded training_results.json  "
              f"(val={training_results.get('val_metrics', {})}, "
              f"test={training_results.get('test_metrics', {})})")

    # ── Step 1: Load session modules ─────────────────────────────────────
    print(f"\n{BOLD}[1/4] Loading session modules…{RESET}")
    GenDataset, GenTask = load_session_modules(session_path)

    gen_dataset   = GenDataset(csv_dir=csv_dir)
    gen_task      = GenTask(gen_dataset)
    db            = gen_dataset.get_db()
    entity_table  = gen_task.entity_table

    # ── Resolve model config ─────────────────────────────────────────────
    cfg = load_model_config(session_path, gen_task)

    if args.show_config:
        print(f"\n{BOLD}Resolved model config (source: {cfg['_source']}){RESET}")
        for k, v in cfg.items():
            if not k.startswith("_"):
                print(f"  {k:<25} = {v}")
        sys.exit(0)

    print(f"  Config source: {cfg['_source']}")
    print(f"  num_layers={cfg['num_layers']}  channels={cfg['channels']}  "
          f"out_channels={cfg['out_channels']}")
    print(f"  num_neighbors={cfg['num_neighbors']}  batch_size={cfg['batch_size']}")

    # ── Get GenTask tables ───────────────────────────────────────────────
    gen_tables = {}
    for split in splits:
        gen_tables[split] = gen_task.get_table(split, mask_input_cols=False)

    # ── Step 2: Fairness check ───────────────────────────────────────────
    author_dataset = AuthorDatasetClass()
    author_task    = AuthorTaskClass(author_dataset)

    fairness_summary = {}
    if not args.skip_fairness_check:
        fairness_summary = run_fairness_check(
            gen_dataset, gen_task,
            author_dataset, author_task,
            cfg,
            author_tables_dir=args.author_tables_dir,
            dataset_name=args.dataset,
            task_name=args.task,
            splits=splits,
        )

    # ── Step 3: Build graph and model ────────────────────────────────────
    print(f"\n{BOLD}[2/4] Building graph and loading model…{RESET}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Text embedder (GloVe, with graceful fallback)
    text_embedder_cfg = None
    try:
        from sentence_transformers import SentenceTransformer
        from torch import Tensor
        from torch_frame.config.text_embedder import TextEmbedderConfig

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
        print("  Text embeddings not available — using None")

    col_to_stype_dict = get_stype_proposal(db)

    # Convert text columns to categorical when no embedder available
    if text_embedder_cfg is None:
        from torch_frame import stype
        for tname in col_to_stype_dict:
            for col in list(col_to_stype_dict[tname]):
                if col_to_stype_dict[tname][col] == stype.text_embedded:
                    print(f"  Converting {tname}.{col}: text → categorical (no embedder)")
                    col_to_stype_dict[tname][col] = stype.categorical

    cache_dir = os.path.join(session_path, "cache")
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=cache_dir,
    )

    # Build loaders — use hyperparams from config, not hardcoded values
    num_neighbors_list = [cfg["num_neighbors"]] * cfg["num_layers"]
    loader_dict = {}
    for split in splits:
        table_input = get_node_train_table_input(
            table=gen_tables[split], task=gen_task
        )
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=num_neighbors_list,
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=cfg["batch_size"],
            temporal_strategy=cfg["temporal_strategy"],
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )

    # Build model from config
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=cfg["num_layers"],
        channels=cfg["channels"],
        out_channels=cfg["out_channels"],
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  Model loaded from {model_path}")

    # ── Step 4: Inference ────────────────────────────────────────────────
    print(f"\n{BOLD}[3/4] Running inference…{RESET}")
    gen_predictions = {}
    gen_metrics     = {}
    for split in splits:
        pred = run_inference(model, loader_dict[split], device, entity_table)
        gen_predictions[split] = pred
        if len(pred) > 0:
            gen_metrics[split] = gen_task.evaluate(pred, gen_tables[split])
            print(f"  [{split}] GenTask metrics: {gen_metrics[split]} ({len(pred)} rows)")
        else:
            gen_metrics[split] = {}
            print(f"  [{split}] No predictions generated")

    # ── Step 5: Compare with author tables ───────────────────────────────
    print(f"\n{BOLD}[4/4] Comparing with author tables…{RESET}")
    comparison_results = {}

    for split in splits:
        author_parquet = os.path.join(
            args.author_tables_dir, args.dataset, args.task, f"{split}.parquet"
        )
        if not os.path.exists(author_parquet):
            print(f"  [{split}] Author table not found: {author_parquet}")
            print(f"         Run: python scripts/download_author_tables.py "
                  f"--dataset {args.dataset} --task {args.task}")
            comparison_results[split] = {"error": "Author table not found"}
            continue

        author_table = Table.load(author_parquet)
        gen_table    = gen_tables[split]
        pred         = gen_predictions[split]

        if len(pred) == 0:
            comparison_results[split] = {"error": "No predictions available"}
            continue

        # Row alignment: merge on (entity, time)
        gen_df          = gen_table.df.copy()
        gen_df["__p__"] = pred
        author_df       = author_table.df.copy()

        gen_ec   = gen_task.entity_col;   gen_tc   = gen_task.time_col
        auth_ec  = author_task.entity_col; auth_tc = author_task.time_col

        merged = pd.merge(
            gen_df[[gen_ec, gen_tc, "__p__"]].rename(
                columns={gen_ec: "_e", gen_tc: "_t"}),
            author_df.rename(columns={auth_ec: "_e", auth_tc: "_t"}),
            on=["_e", "_t"], how="inner",
        )

        matched_rows  = len(merged)
        gen_rows      = len(gen_df)
        author_rows   = len(author_df)

        author_metrics_split = {}
        if matched_rows > 0:
            matched_author_df = merged.rename(
                columns={"_e": auth_ec, "_t": auth_tc}
            )
            author_table_cols = list(author_table.df.columns)
            matched_author_table = Table(
                df=matched_author_df[author_table_cols],
                fkey_col_to_pkey_table=author_table.fkey_col_to_pkey_table,
                pkey_col=author_table.pkey_col,
                time_col=author_table.time_col,
            )
            matched_pred = merged["__p__"].to_numpy()
            try:
                author_metrics_split = author_task.evaluate(matched_pred, matched_author_table)
            except Exception as e:
                author_metrics_split = {"error": str(e)}
                print(f"  [{split}] Author evaluation error: {e}")

        # Coverage warning
        cov = matched_rows / author_rows if author_rows else 0
        if 0 < cov < 0.70:
            print(f"  {YELLOW}⚠️  [{split}] Only {cov:.1%} of author rows matched "
                  f"— author_metrics may be unreliable.{RESET}")

        # Print comparison table
        clean_author_m = (
            author_metrics_split
            if isinstance(author_metrics_split, dict)
            and "error" not in author_metrics_split
            else {}
        )
        print(format_table(split, gen_metrics.get(split, {}), clean_author_m,
                           gen_rows, author_rows, matched_rows))

        comparison_results[split] = {
            "gen_metrics":    {k: float(v) for k, v in gen_metrics.get(split, {}).items()},
            "author_metrics": (
                {k: float(v) for k, v in author_metrics_split.items()}
                if isinstance(author_metrics_split, dict)
                and "error" not in author_metrics_split
                else author_metrics_split
            ),
            "gen_rows":     gen_rows,
            "author_rows":  author_rows,
            "matched_rows": matched_rows,
            "coverage":     round(cov, 4),
        }

    # ── Output JSON ──────────────────────────────────────────────────────
    output_data = {
        "session":          args.session,
        "dataset":          args.dataset,
        "task":             args.task,
        "model_config":     {k: v for k, v in cfg.items() if not k.startswith("_")},
        "fairness_summary": fairness_summary,
        "splits":           comparison_results,
        "training_results": training_results,
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nComparison saved to {args.output}")
    else:
        print(f"\n--- JSON Output ---")
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
