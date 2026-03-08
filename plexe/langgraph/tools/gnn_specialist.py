from typing import Dict, Any
from langchain_core.tools import tool as langchain_tool

@langchain_tool
def generate_training_script(
    dataset_module_path: str,
    dataset_class_name: str,
    task_module_path: str,
    task_class_name: str,
    working_dir: str,
    csv_dir: str = None,
    task_type: str = "regression",
    tune_metric: str = "mae",
    higher_is_better: bool = False,
    out_channels: int = 1,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 0.005,
    hidden_channels: int = 128,
    num_gnn_layers: int = 2,
) -> Dict[str, Any]:
    """
    Generate a GNN training script using plexe.relbench.modeling modules.
    
    Args:
        dataset_module_path: Path to the Dataset Python module
        dataset_class_name: Name of the Dataset class
        task_module_path: Path to the Task Python module
        task_class_name: Name of the Task class
        working_dir: Working directory for outputs
        csv_dir: Path to CSV files directory (defaults to working_dir/csv_files)
        task_type: Type of task (regression, binary_classification, multiclass_classification)
        tune_metric: Metric to optimize
        higher_is_better: Whether higher metric values are better
        out_channels: Output channels for the model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        hidden_channels: Hidden channels in GNN
        num_gnn_layers: Number of GNN layers
    
    Returns:
        Path to generated script
    """
    import os

    # Normalize common metric aliases to actual function names in plexe.relbench.metrics
    METRIC_NAME_MAP = {
        "auroc": "roc_auc",
        "auc": "roc_auc",
        "roc_auc_score": "roc_auc",
        "ap": "average_precision",
        "mean_absolute_error": "mae",
        "mean_squared_error": "mse",
        "root_mean_squared_error": "rmse",
        "r2_score": "r2",
        "f1_score": "f1",
        "map": "link_prediction_map",
        "ndcg": "link_prediction_ndcg",
    }
    tune_metric = METRIC_NAME_MAP.get(tune_metric.lower(), tune_metric)

    working_dir = os.path.abspath(working_dir)
    
    # Use csv_dir from parameter or default to working_dir/csv_files
    if csv_dir is None:
        csv_dir = f"{working_dir}/csv_files"
    else:
        csv_dir = os.path.abspath(csv_dir)
    
    script_template = f'''"""
Auto-generated GNN training script using plexe.relbench.modeling.
Faithfully follows the reference RelBench training notebook.
"""

import os
import sys
import copy
import math
import json
import numpy as np
from tqdm import tqdm

import torch

sys.path.insert(0, "{os.path.dirname(dataset_module_path)}")
sys.path.insert(0, "{os.path.dirname(task_module_path)}")

from dataset import {dataset_class_name}
from task import {task_class_name}

from plexe.relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from plexe.relbench.modeling.utils import get_stype_proposal
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# --- Text Embedding (GloVe, with fallback to None) ---
text_embedder_cfg = None
try:
    from typing import List, Optional as Opt
    from sentence_transformers import SentenceTransformer
    from torch import Tensor
    from torch_frame.config.text_embedder import TextEmbedderConfig

    class GloveTextEmbedding:
        def __init__(self, device: Opt[torch.device] = None):
            self.model = SentenceTransformer(
                "sentence-transformers/average_word_embeddings_glove.6B.300d",
                device=device,
            )
        def __call__(self, sentences: List[str]) -> Tensor:
            return torch.from_numpy(self.model.encode(sentences))

    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    )
    print("Using GloVe text embeddings")
except Exception as e:
    print(f"Text embedding not available ({{e}}), using None")

# --- Dataset & Task ---
csv_dir = "{csv_dir}"
dataset = {dataset_class_name}(csv_dir=csv_dir)
task = {task_class_name}(dataset)
db = dataset.get_db()

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")

print(f"Train samples: {{len(train_table)}}")
print(f"Val samples: {{len(val_table)}}")
print(f"Test samples: {{len(test_table)}}")

# --- Build Graph (stays on CPU; only batches move to GPU) ---
col_to_stype_dict = get_stype_proposal(db)
data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=text_embedder_cfg,
    cache_dir="{working_dir}/cache/",
)

entity_table = task.entity_table

# --- Data Loaders ---
loader_dict = {{}}
for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
    table_input = get_node_train_table_input(table=table, task=task)
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[128] * {num_gnn_layers},
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size={batch_size},
        temporal_strategy="uniform",
        shuffle=(split == "train"),
        num_workers=0,
        persistent_workers=False,
    )

# --- Model (matches reference notebook exactly) ---
class Model(torch.nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels, out_channels):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={{
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            }},
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

model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers={num_gnn_layers},
    channels={hidden_channels},
    out_channels={out_channels},
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr={learning_rate})

# --- Loss ---
task_type = "{task_type}"
if task_type == "binary_classification":
    loss_fn = torch.nn.BCEWithLogitsLoss()
elif task_type == "multiclass_classification":
    loss_fn = torch.nn.CrossEntropyLoss()
else:
    loss_fn = torch.nn.L1Loss()

# --- Metric config ---
tune_metric = "{tune_metric}"
higher_is_better = {higher_is_better}

def _get_metric(metrics_dict, metric_name):
    """Get metric value with alias fallback."""
    if metric_name in metrics_dict:
        v = metrics_dict[metric_name]
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    # Alias fallback
    aliases = {{
        "auroc": "roc_auc", "roc_auc": "auroc", "auc": "roc_auc",
        "ap": "average_precision", "average_precision": "ap",
    }}
    alt = aliases.get(metric_name)
    if alt and alt in metrics_dict:
        v = metrics_dict[alt]
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            print(f"  Note: '{{metric_name}}' not found, using '{{alt}}' instead")
            return v
    return None

# --- Training loop ---
state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf

for epoch in range(1, {epochs} + 1):
    # Train
    model.train()
    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"], desc=f"Epoch {{epoch}}"):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        target = batch[entity_table].y
        if task_type == "multiclass_classification":
            target = target.long()
        else:
            target = target.float()

        loss = loss_fn(pred.float(), target)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    train_loss = loss_accum / count_accum

    # Validate using task.evaluate()
    model.eval()
    val_pred_list = []
    with torch.no_grad():
        for batch in loader_dict["val"]:
            batch = batch.to(device)
            pred = model(batch, entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            val_pred_list.append(pred.detach().cpu())

    val_pred = torch.cat(val_pred_list, dim=0).numpy()
    val_metrics = task.evaluate(val_pred, val_table)
    print(f"Epoch {{epoch}}/{epochs}: Loss={{train_loss:.4f}}, Val metrics: {{val_metrics}}")

    current_val = _get_metric(val_metrics, tune_metric)
    if current_val is None:
        print(f"  Warning: metric '{{tune_metric}}' unavailable (got {{list(val_metrics.keys())}}), saving model by loss")
        state_dict = copy.deepcopy(model.state_dict())
        continue

    if (higher_is_better and current_val > best_val_metric) or (
        not higher_is_better and current_val < best_val_metric
    ):
        best_val_metric = current_val
        state_dict = copy.deepcopy(model.state_dict())
        print(f"  -> New best model! {{tune_metric}}={{best_val_metric:.4f}}")

# --- Final Evaluation ---
assert state_dict is not None, "No best model found during training"
model.load_state_dict(state_dict)
model.eval()

# Best-val re-evaluation
val_pred_list = []
with torch.no_grad():
    for batch in loader_dict["val"]:
        batch = batch.to(device)
        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        val_pred_list.append(pred.detach().cpu())

val_pred = torch.cat(val_pred_list, dim=0).numpy()
val_metrics = task.evaluate(val_pred, val_table)
print(f"\\nBest Val metrics: {{val_metrics}}")

# Test evaluation
test_pred_list = []
with torch.no_grad():
    for batch in loader_dict["test"]:
        batch = batch.to(device)
        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        test_pred_list.append(pred.detach().cpu())

test_pred = torch.cat(test_pred_list, dim=0).numpy()
test_metrics = task.evaluate(test_pred)
print(f"Best Test metrics: {{test_metrics}}")

# --- Save model & results ---
torch.save(state_dict, "{working_dir}/best_model.pt")

def _to_json_safe(d):
    """Convert numpy/torch values to Python floats for JSON serialization."""
    return {{k: float(v) for k, v in d.items()}}

results = {{
    "best_val_{tune_metric}": float(best_val_metric),
    "tune_metric": tune_metric,
    "val_metrics": _to_json_safe(val_metrics),
    "test_metrics": _to_json_safe(test_metrics),
    "model_path": "{working_dir}/best_model.pt",
    "epochs_trained": {epochs},
}}

with open("{working_dir}/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\\nTraining complete! Results saved to {working_dir}/training_results.json")
'''
    
    # Use the working_dir directly - it's passed from the state
    script_path = os.path.join(working_dir, "train_script.py")
    os.makedirs(working_dir, exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_template)
    
    return {
        "status": "generated",
        "script_path": script_path,
    }


@langchain_tool
def execute_training_script(
    script_path: str,
    timeout: int = 3600
) -> Dict[str, Any]:
    """
    Execute a training script.
    
    Args:
        script_path: Path to the training script
        timeout: Maximum execution time in seconds
    
    Returns:
        Execution results
    """
    import subprocess
    import os
    import json
    import re

    script_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_path)

    _patch_broken_imports(script_path, script_dir)

    env = {**os.environ}
    existing = env.get("PYTHONPATH", "")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    paths = [script_dir, project_root]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    try:
        result = subprocess.run(
            ["python", script_path],
            cwd=script_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        results_path = os.path.join(script_dir, "training_results.json")

        training_results = {}
        if os.path.exists(results_path):
            with open(results_path) as f:
                training_results = json.load(f)

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "training_results": training_results
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f"Script execution exceeded {timeout} seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def _patch_broken_imports(script_path: str, script_dir: str):
    """Fix dotted-path imports that reference the workdir directory structure.

    LLMs sometimes generate imports like:
        from workdir.session_xxx.dataset import GenDataset
    instead of:
        from dataset import GenDataset

    This rewrites those lines in-place so the script can run.
    """
    import re

    try:
        with open(script_path, 'r') as f:
            content = f.read()

        pattern = re.compile(
            r'^(\s*from\s+)'
            r'(?:workdir|session)[.\w]*\.'
            r'(dataset|task)'
            r'(\s+import\s+.+)$',
            re.MULTILINE,
        )

        patched = pattern.sub(r'\1\2\3', content)

        if patched != content:
            with open(script_path, 'w') as f:
                f.write(patched)
    except Exception:
        pass

