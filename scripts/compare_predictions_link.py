#!/usr/bin/env python3
"""
Re-run link-prediction model inference from a workdir session and compare
recommendations against author ground truth tables.  Includes an integrated
fairness check that verifies timestamp alignment, task-definition alignment,
hyperparameter consistency, and matched-row coverage before showing the
metric comparison.

This is the **recommendation / link-prediction** counterpart of
``compare_predictions.py`` (which handles entity classification / regression).

Usage:
    python scripts/compare_predictions_link.py \
        --session session-abc123 \
        --dataset rel-hm --task user-item-purchase

    python scripts/compare_predictions_link.py \
        --session session-abc123 \
        --dataset rel-amazon --task user-item-rate \
        --output results/amazon_rate.json

    # Skip the fairness banner (e.g. in CI)
    python scripts/compare_predictions_link.py ... --skip-fairness-check

    # Just print the hyper-param config that will be used
    python scripts/compare_predictions_link.py ... --show-config
"""

import argparse
import importlib
import json
import os
import re
import sys
import types

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import MLP

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from plexe.relbench.base import EntityTask, RecommendationTask, Table, TaskType
from plexe.relbench.modeling.graph import make_pkey_fkey_graph
from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from plexe.relbench.modeling.utils import get_stype_proposal, to_unix_time
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


# ── Two-tower link-prediction model (mirrors gnn_link.py) ────────────────────
class ShallowEmbedding(torch.nn.Module):
    """Per-node learnable embedding for ID-aware link prediction."""

    def __init__(self, num_nodes: int, channels: int):
        super().__init__()
        self.emb = torch.nn.Embedding(num_nodes, channels)
        torch.nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)

    def reset_parameters(self):
        torch.nn.init.normal_(self.emb.weight, std=0.1)


class IdGNNModel(torch.nn.Module):
    """ID-GNN model for link prediction (per-source scoring)."""

    def __init__(self, data, col_stats_dict, num_layers, channels,
                 num_src_nodes, num_dst_nodes):
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
        self.head = MLP(channels, out_channels=num_dst_nodes,
                        norm="batch_norm", num_layers=1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch, src_entity_table):
        seed_time = batch[src_entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for nt, rt in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rt
        x_dict = self.gnn(
            x_dict, batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        return self.head(x_dict[src_entity_table][:seed_time.size(0)])


class TwoTowerModel(torch.nn.Module):
    """Two-tower GNN model for link prediction (inner-product scoring)."""

    def __init__(self, data, col_stats_dict, num_layers, channels):
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
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for nt, rt in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rt
        x_dict = self.gnn(
            x_dict, batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        return x_dict[entity_table][:seed_time.size(0)]


# ── Hyper-param config reader ──────────────────────────────────────────────────
_AUTHOR_DEFAULTS = {
    "num_layers":        2,
    "channels":          128,
    "num_neighbors":     128,
    "batch_size":        512,
    "temporal_strategy": "uniform",
    "model_type":        "idgnn",    # "idgnn" or "twotower"
}

_RE_NUM_NEIGHBORS = re.compile(r"num_neighbors\s*=\s*\[(\d+)\]\s*\*\s*(\d+)")
_RE_BATCH_SIZE    = re.compile(r"batch_size\s*=\s*(\d+)")
_RE_NUM_LAYERS    = re.compile(r"num_layers\s*=\s*(\d+)")
_RE_CHANNELS      = re.compile(r"\bchannels\s*=\s*(\d+)")
_RE_MODEL_TYPE    = re.compile(r'model_type\s*=\s*["\'](\w+)["\']')
_RE_MODEL_INIT    = re.compile(r"Model\s*\(")
_RE_LOADER_INIT   = re.compile(r"(?:NeighborLoader|LinkNeighborLoader)\s*\(")


def load_model_config(session_path: str, gen_task=None) -> dict:
    """
    Read hyper-parameters in priority order:
      1. session/model_config.json
      2. session/train_script.py (regex parsing)
      3. Author defaults
    """
    cfg = dict(_AUTHOR_DEFAULTS)
    source = "author defaults"

    config_path = os.path.join(session_path, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        cfg.update({k: v for k, v in saved.items() if k in cfg})
        source = "model_config.json"
    else:
        ts_path = os.path.join(session_path, "train_script.py")
        if os.path.exists(ts_path):
            text = open(ts_path).read()

            got_layers_from_neighbors = False
            m = _RE_NUM_NEIGHBORS.search(text)
            if m:
                cfg["num_neighbors"] = int(m.group(1))
                cfg["num_layers"] = int(m.group(2))
                got_layers_from_neighbors = True

            loader_pos = _RE_LOADER_INIT.search(text)
            loader_offset = loader_pos.start() if loader_pos else 0
            best_bs, best_dist = None, None
            for bm in _RE_BATCH_SIZE.finditer(text):
                if bm.start() >= loader_offset and loader_pos:
                    dist = bm.start() - loader_offset
                    if best_dist is None or dist < best_dist:
                        best_bs, best_dist = int(bm.group(1)), dist
            if best_bs is not None:
                cfg["batch_size"] = best_bs
            else:
                m = _RE_BATCH_SIZE.search(text)
                if m:
                    cfg["batch_size"] = int(m.group(1))

            if not got_layers_from_neighbors:
                model_pos = _RE_MODEL_INIT.search(text)
                model_offset = model_pos.start() if model_pos else 0
                best_nl, best_dist = None, None
                for nm in _RE_NUM_LAYERS.finditer(text):
                    if nm.start() >= model_offset and model_pos:
                        dist = nm.start() - model_offset
                        if best_dist is None or dist < best_dist:
                            best_nl, best_dist = int(nm.group(1)), dist
                if best_nl is not None:
                    cfg["num_layers"] = best_nl
                else:
                    m = _RE_NUM_LAYERS.search(text)
                    if m:
                        cfg["num_layers"] = int(m.group(1))

            m = _RE_CHANNELS.search(text)
            if m:
                cfg["channels"] = int(m.group(1))
            m = _RE_MODEL_TYPE.search(text)
            if m:
                cfg["model_type"] = m.group(1)
            source = "train_script.py (parsed)"

    if gen_task is not None and isinstance(gen_task, RecommendationTask):
        cfg["num_dst_nodes"] = gen_task.num_dst_nodes
        cfg["num_src_nodes"] = gen_task.num_src_nodes
        cfg["eval_k"] = gen_task.eval_k

    cfg["_source"] = source
    return cfg


# ── Session module loader ──────────────────────────────────────────────────────
def load_session_modules(session_path):
    """Dynamically import GenDataset and GenTask from a session directory."""
    if session_path not in sys.path:
        sys.path.insert(0, session_path)
    folder_name    = os.path.basename(session_path)
    workdir_parent = os.path.dirname(os.path.dirname(session_path))
    if workdir_parent not in sys.path:
        sys.path.insert(0, workdir_parent)
    try:
        dm = importlib.import_module(f"workdir.{folder_name}.dataset")
        GenDataset = dm.GenDataset
    except Exception as e:
        raise RuntimeError(
            f"Cannot import GenDataset from {session_path}/dataset.py: {e}"
        )
    try:
        tm = importlib.import_module(f"workdir.{folder_name}.task")
        GenTask = tm.GenTask
    except Exception as e:
        raise RuntimeError(
            f"Cannot import GenTask from {session_path}/task.py: {e}"
        )
    return GenDataset, GenTask


# ── Link-prediction inference ─────────────────────────────────────────────────
def run_link_inference_idgnn(model, loader, device, src_entity_table,
                             num_dst_nodes, eval_k):
    """Run inference with ID-GNN model: output is (N, eval_k) top-k dst indices."""
    model.eval()
    all_topk = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch, src_entity_table)
            topk = logits.topk(eval_k, dim=1).indices
            all_topk.append(topk.cpu())
    if not all_topk:
        return np.array([]).reshape(0, eval_k)
    return torch.cat(all_topk, dim=0).numpy()


def run_link_inference_twotower(src_model, loader, device,
                                src_entity_table, dst_entity_table,
                                dst_emb, eval_k):
    """Run inference with two-tower model: inner-product scoring."""
    src_model.eval()
    all_topk = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            src_embs = src_model(batch, src_entity_table)
            scores = src_embs @ dst_emb.T
            topk = scores.topk(eval_k, dim=1).indices
            all_topk.append(topk.cpu())
    if not all_topk:
        return np.array([]).reshape(0, eval_k)
    return torch.cat(all_topk, dim=0).numpy()


# ── Fairness check ─────────────────────────────────────────────────────────────
def run_fairness_check(gen_dataset, gen_task,
                       author_dataset, author_task,
                       cfg: dict,
                       author_tables_dir: str,
                       dataset_name: str, task_name: str,
                       splits: list) -> dict:
    _hdr("Fairness Check (Link Prediction)")
    summary = {"timestamp_ok": True, "task_ok": True, "coverage": {}}

    # 1. Timestamp alignment
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

    # 2. Task-definition alignment (link-prediction specific fields)
    print()
    task_fields = [
        "src_entity_col", "src_entity_table",
        "dst_entity_col", "dst_entity_table",
        "time_col", "timedelta", "task_type", "eval_k",
    ]
    for field in task_fields:
        gv = str(getattr(gen_task,    field, "N/A"))
        av = str(getattr(author_task, field, "N/A"))
        if gv == av:
            _ok(f"{field:<24} = {gv}")
        else:
            _fail(f"{field:<24}: GenTask={gv!r}  AuthorTask={av!r}")
            summary["task_ok"] = False

    if not summary["task_ok"]:
        print(f"\n  {RED}→ Task-definition mismatch: metrics will measure DIFFERENT things.{RESET}")

    # 3. Hyperparameter transparency
    print(f"\n  Hyperparams (source: {BOLD}{cfg['_source']}{RESET})")
    _info(f"num_layers={cfg['num_layers']}  channels={cfg['channels']}  "
          f"model_type={cfg['model_type']}")
    _info(f"num_neighbors={cfg['num_neighbors']}  batch_size={cfg['batch_size']}  "
          f"temporal_strategy={cfg['temporal_strategy']}")

    ref_nn = 128
    if cfg["num_neighbors"] != ref_nn:
        _warn(f"num_neighbors={cfg['num_neighbors']} differs from author ref ({ref_nn})")
    else:
        _ok(f"num_neighbors matches author reference ({ref_nn})")

    # 4. Matched-row coverage
    print()
    src_col_gen  = gen_task.src_entity_col
    src_col_auth = author_task.src_entity_col
    time_col_gen  = gen_task.time_col
    time_col_auth = author_task.time_col
    for split in splits:
        parquet = os.path.join(
            author_tables_dir, dataset_name, task_name, f"{split}.parquet"
        )
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
            gen_tbl.df[[src_col_gen, time_col_gen]].rename(
                columns={src_col_gen: "_src", time_col_gen: "_t"}),
            author_tbl.df[[src_col_auth, time_col_auth]].rename(
                columns={src_col_auth: "_src", time_col_auth: "_t"}),
            on=["_src", "_t"], how="inner",
        )
        gen_n   = len(gen_tbl.df)
        auth_n  = len(author_tbl.df)
        match_n = len(merged)
        cov     = match_n / auth_n if auth_n else 0
        summary["coverage"][split] = cov

        cov_str = f"{cov:.1%}  ({match_n:,} / {auth_n:,} author rows)"
        if cov >= 0.90:   _ok(f"[{split}] Coverage {cov_str}")
        elif cov >= 0.70: _warn(f"[{split}] Coverage {cov_str} — some noise in author_metrics")
        else:             _fail(f"[{split}] Coverage {cov_str} — author_metrics UNRELIABLE")

    return summary


# ── Comparison table printer ───────────────────────────────────────────────────
def format_table(split, gen_metrics, author_metrics, gen_rows, author_rows, matched_rows):
    all_m = sorted(set(list(gen_metrics) + list(author_metrics)))
    w1, w2, w3 = 30, 14, 14
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


# ── Verdict ────────────────────────────────────────────────────────────────────
def _compute_verdict(fairness_summary: dict, comparison_results: dict,
                     splits: list) -> dict:
    """Synthesize all signals into a trustworthiness verdict."""
    issues = []
    warnings = []

    if fairness_summary:
        if not fairness_summary.get("timestamp_ok", True):
            issues.append("Timestamp mismatch — train/val/test splits differ between GenTask and AuthorTask")
        if not fairness_summary.get("task_ok", True):
            warnings.append("Task-definition mismatch (src/dst entity cols, timedelta, or eval_k)")

        for split, cov in fairness_summary.get("coverage", {}).items():
            if cov is None:
                warnings.append(f"[{split}] Author table missing — cannot compute author metrics")
            elif cov < 0.50:
                issues.append(f"[{split}] Coverage {cov:.0%} — too few matched rows, author metrics unreliable")
            elif cov < 0.70:
                warnings.append(f"[{split}] Coverage {cov:.0%} — borderline, author metrics may have noise")

    for split in splits:
        sr = comparison_results.get(split, {})
        if isinstance(sr, dict) and "error" in sr:
            issues.append(f"[{split}] {sr['error']}")
            continue
        matched = sr.get("matched_rows", 0)
        author  = sr.get("author_rows", 0)
        if author > 0 and matched == 0:
            issues.append(f"[{split}] Zero matched rows — comparison is meaningless")
        am = sr.get("author_metrics", {})
        if isinstance(am, dict) and "error" in am:
            issues.append(f"[{split}] Author evaluation failed: {am['error']}")

    if issues:
        trust = "UNRELIABLE"
    elif warnings:
        trust = "TRUSTWORTHY_WITH_CAVEATS"
    else:
        trust = "TRUSTWORTHY"

    return {"trust_level": trust, "issues": issues, "warnings": warnings}


def _print_verdict(verdict: dict, dataset: str, task: str):
    trust = verdict["trust_level"]
    issues = verdict["issues"]
    warnings = verdict["warnings"]

    _hdr(f"Verdict: {dataset} / {task}")

    if trust == "TRUSTWORTHY":
        print(f"\n  {GREEN}{BOLD}✅ TRUSTWORTHY{RESET}")
        print(f"  {GREEN}The comparison results can be relied upon.")
        print(f"  Timestamps match, task definitions align, and coverage is sufficient.{RESET}")
    elif trust == "TRUSTWORTHY_WITH_CAVEATS":
        print(f"\n  {YELLOW}{BOLD}⚠️  TRUSTWORTHY WITH CAVEATS{RESET}")
        print(f"  {YELLOW}The comparison is usable but has minor concerns:{RESET}")
        for w in warnings:
            print(f"    {YELLOW}• {w}{RESET}")
        print(f"\n  {YELLOW}GenTask metrics are reliable.")
        print(f"  Author metrics may have minor deviations due to the above.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}❌ UNRELIABLE{RESET}")
        print(f"  {RED}The comparison should NOT be trusted due to:{RESET}")
        for i in issues:
            print(f"    {RED}• {i}{RESET}")
        if warnings:
            print(f"  {YELLOW}Additional warnings:{RESET}")
            for w in warnings:
                print(f"    {YELLOW}• {w}{RESET}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compare link-prediction model against author ground truth (with fairness check)"
    )
    parser.add_argument("--session",  required=True,
                        help="Session folder name or full path")
    parser.add_argument("--dataset",  required=True,
                        help="Dataset name (e.g. rel-hm)")
    parser.add_argument("--task",     required=True,
                        help="Task name (e.g. user-item-purchase)")
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

    # ── Guard: entity tasks need a different script ───────────────────────
    if not issubclass(AuthorTaskClass, RecommendationTask):
        _entity_tasks = [
            t for t, cls in ds_info["tasks"].items()
            if not issubclass(cls, RecommendationTask)
        ]
        _link_tasks = [
            t for t, cls in ds_info["tasks"].items()
            if issubclass(cls, RecommendationTask)
        ]
        print(
            f"{RED}Error: '{args.task}' is an entity-level task (classification / regression).\n"
            f"  This script only supports recommendation (link-prediction) tasks.\n"
            f"  Use  scripts/compare_predictions.py  instead for entity tasks.\n\n"
            f"  Entity tasks for '{args.dataset}':        {_entity_tasks}\n"
            f"  Recommendation tasks for '{args.dataset}': {_link_tasks}{RESET}"
        )
        sys.exit(1)

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

    gen_dataset = GenDataset(csv_dir=csv_dir)
    gen_task    = GenTask(gen_dataset)
    db          = gen_dataset.get_db()

    src_entity_table = gen_task.src_entity_table
    dst_entity_table = gen_task.dst_entity_table
    eval_k           = gen_task.eval_k
    num_dst_nodes    = gen_task.num_dst_nodes

    # ── Resolve model config ─────────────────────────────────────────────
    cfg = load_model_config(session_path, gen_task)

    if args.show_config:
        print(f"\n{BOLD}Resolved model config (source: {cfg['_source']}){RESET}")
        for k, v in cfg.items():
            if not k.startswith("_"):
                print(f"  {k:<25} = {v}")
        sys.exit(0)

    print(f"  Config source: {cfg['_source']}")
    print(f"  model_type={cfg['model_type']}  num_layers={cfg['num_layers']}  "
          f"channels={cfg['channels']}")
    print(f"  num_neighbors={cfg['num_neighbors']}  batch_size={cfg['batch_size']}  "
          f"eval_k={eval_k}")

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

    num_neighbors_list = [cfg["num_neighbors"]] * cfg["num_layers"]

    # Build link-prediction loaders using NeighborLoader on src nodes
    from torch_geometric.loader import NeighborLoader

    loader_dict = {}
    for split in splits:
        table = gen_tables[split]
        src_nodes = torch.from_numpy(
            table.df[gen_task.src_entity_col].astype(int).values
        )
        src_time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=num_neighbors_list,
            time_attr="time",
            input_nodes=(src_entity_table, src_nodes),
            input_time=src_time,
            batch_size=cfg["batch_size"],
            temporal_strategy=cfg["temporal_strategy"],
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )

    # Build model
    model_type = cfg["model_type"]
    if model_type == "idgnn":
        model = IdGNNModel(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=cfg["num_layers"],
            channels=cfg["channels"],
            num_src_nodes=gen_task.num_src_nodes,
            num_dst_nodes=num_dst_nodes,
        ).to(device)
    elif model_type == "twotower":
        model = TwoTowerModel(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=cfg["num_layers"],
            channels=cfg["channels"],
        ).to(device)
    else:
        print(f"Error: Unknown model_type '{model_type}'. Must be 'idgnn' or 'twotower'.")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  Model loaded from {model_path}  (type={model_type})")

    # ── Step 4: Inference ────────────────────────────────────────────────
    print(f"\n{BOLD}[3/4] Running link-prediction inference…{RESET}")
    gen_predictions = {}
    gen_metrics     = {}

    for split in splits:
        if model_type == "idgnn":
            pred = run_link_inference_idgnn(
                model, loader_dict[split], device,
                src_entity_table, num_dst_nodes, eval_k,
            )
        else:
            # Two-tower: need all dst embeddings first
            dst_emb = model.encoder(data.tf_dict)[dst_entity_table].to(device)
            pred = run_link_inference_twotower(
                model, loader_dict[split], device,
                src_entity_table, dst_entity_table,
                dst_emb, eval_k,
            )
        gen_predictions[split] = pred
        if pred.shape[0] > 0:
            gen_metrics[split] = gen_task.evaluate(pred, gen_tables[split])
            print(f"  [{split}] GenTask metrics: {gen_metrics[split]} ({pred.shape[0]} rows)")
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

        if pred.shape[0] == 0:
            comparison_results[split] = {"error": "No predictions available"}
            continue

        # Row alignment: merge on (src_entity, time)
        gen_df   = gen_table.df.copy()
        gen_src  = gen_task.src_entity_col
        gen_tc   = gen_task.time_col
        auth_src = author_task.src_entity_col
        auth_tc  = author_task.time_col

        gen_df["__pred_idx__"] = range(len(gen_df))

        merged = pd.merge(
            gen_df[[gen_src, gen_tc, "__pred_idx__"]].rename(
                columns={gen_src: "_src", gen_tc: "_t"}),
            author_table.df.reset_index().rename(
                columns={auth_src: "_src", auth_tc: "_t", "index": "_auth_idx"}),
            on=["_src", "_t"], how="inner",
        )

        matched_rows = len(merged)
        gen_rows     = len(gen_df)
        author_rows  = len(author_table.df)

        author_metrics_split = {}
        if matched_rows > 0:
            matched_pred_indices = merged["__pred_idx__"].values
            matched_pred = pred[matched_pred_indices]

            matched_author_df = merged.rename(
                columns={"_src": auth_src, "_t": auth_tc}
            )
            author_table_cols = list(author_table.df.columns)
            matched_author_table = Table(
                df=matched_author_df[author_table_cols].reset_index(drop=True),
                fkey_col_to_pkey_table=author_table.fkey_col_to_pkey_table,
                pkey_col=author_table.pkey_col,
                time_col=author_table.time_col,
            )
            try:
                author_metrics_split = author_task.evaluate(
                    matched_pred, matched_author_table
                )
            except Exception as e:
                author_metrics_split = {"error": str(e)}
                print(f"  [{split}] Author evaluation error: {e}")

        cov = matched_rows / author_rows if author_rows else 0
        if 0 < cov < 0.70:
            print(f"  {YELLOW}⚠️  [{split}] Only {cov:.1%} of author rows matched "
                  f"— author_metrics may be unreliable.{RESET}")

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

    # ── Final verdict ────────────────────────────────────────────────────
    verdict = _compute_verdict(fairness_summary, comparison_results, splits)
    _print_verdict(verdict, args.dataset, args.task)

    # ── Output JSON ──────────────────────────────────────────────────────
    output_data = {
        "session":          args.session,
        "dataset":          args.dataset,
        "task":             args.task,
        "task_type":        "link_prediction",
        "model_config":     {k: v for k, v in cfg.items() if not k.startswith("_")},
        "fairness_summary": fairness_summary,
        "splits":           comparison_results,
        "training_results": training_results,
        "verdict":          verdict,
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
