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
    
    # Use csv_dir from parameter or default to working_dir/csv_files
    if csv_dir is None:
        csv_dir = f"{working_dir}/csv_files"
    
    script_template = f'''"""
Auto-generated GNN training script using plexe.relbench.modeling.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import Adam
from datetime import datetime

sys.path.insert(0, "{os.path.dirname(dataset_module_path)}")
sys.path.insert(0, "{os.path.dirname(task_module_path)}")

from dataset import {dataset_class_name}
from task import {task_class_name}

from plexe.relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from plexe.relbench.modeling.utils import get_stype_proposal
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

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

col_to_stype_dict = get_stype_proposal(db)
data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=None,
    cache_dir="{working_dir}/cache/",
)

data = data.to(device)
entity_table = task.entity_table

def create_loader(table, shuffle=False):
    table_input = get_node_train_table_input(table=table, task=task)
    return NeighborLoader(
        data,
        num_neighbors=[128] * {num_gnn_layers},
        time_attr="time",
        input_nodes=(entity_table, table_input.nodes[entity_table]),
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size={batch_size},
        temporal_strategy="uniform",
        shuffle=shuffle,
    )

train_loader = create_loader(train_table, shuffle=True)
val_loader = create_loader(val_table)
test_loader = create_loader(test_table)

class GNNModel(torch.nn.Module):
    def __init__(self, data, col_stats_dict, hidden_channels={hidden_channels}, out_channels={out_channels}):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=hidden_channels,
            node_to_col_names={{
                node_type: list(col_stats_dict[node_type].keys())
                for node_type in data.node_types
                if node_type in col_stats_dict
            }},
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=data.node_types,
            channels=hidden_channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=hidden_channels,
            num_layers={num_gnn_layers},
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, out_channels),
        )
    
    def forward(self, batch, entity_table):
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            batch.seed_time, batch.time_dict, batch.batch_dict
        )
        for node_type in x_dict:
            x_dict[node_type] = x_dict[node_type] + rel_time_dict[node_type]
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return self.head(x_dict[entity_table])

model = GNNModel(data, col_stats_dict).to(device)
optimizer = Adam(model.parameters(), lr={learning_rate})

task_type = "{task_type}"
if task_type == "binary_classification":
    loss_fn = torch.nn.BCEWithLogitsLoss()
elif task_type == "multiclass_classification":
    loss_fn = torch.nn.CrossEntropyLoss()
else:
    loss_fn = torch.nn.MSELoss()

best_val_metric = float('inf') if not {str(higher_is_better).lower()} else float('-inf')
best_model_path = "{working_dir}/best_model.pt"

for epoch in range({epochs}):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch, entity_table).squeeze()
        y = batch[entity_table].y.float()
        if task_type == "multiclass_classification":
            y = y.long()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch, entity_table).squeeze()
            val_preds.append(pred.cpu())
            val_labels.append(batch[entity_table].y.cpu())
    
    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)
    
    if task_type == "regression":
        val_metric = F.mse_loss(val_preds, val_labels.float()).sqrt().item()
    elif task_type == "binary_classification":
        val_metric = ((val_preds > 0).float() == val_labels.float()).float().mean().item()
    else:
        val_metric = (val_preds.argmax(dim=-1) == val_labels).float().mean().item()
    
    print(f"Epoch {{epoch+1}}/{epochs}: Loss={{total_loss:.4f}}, Val {tune_metric}={{val_metric:.4f}}")
    
    is_better = val_metric < best_val_metric if not {str(higher_is_better).lower()} else val_metric > best_val_metric
    if is_better:
        best_val_metric = val_metric
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> New best model saved!")

model.load_state_dict(torch.load(best_model_path))
model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch, entity_table).squeeze()
        test_preds.append(pred.cpu())
        test_labels.append(batch[entity_table].y.cpu())

test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)

if task_type == "regression":
    test_metric = F.mse_loss(test_preds, test_labels.float()).sqrt().item()
    print(f"\\nTest RMSE: {{test_metric:.4f}}")
elif task_type == "binary_classification":
    test_metric = ((test_preds > 0).float() == test_labels.float()).float().mean().item()
    print(f"\\nTest Accuracy: {{test_metric:.4f}}")
else:
    test_metric = (test_preds.argmax(dim=-1) == test_labels).float().mean().item()
    print(f"\\nTest Accuracy: {{test_metric:.4f}}")

results = {{
    "best_val_{tune_metric}": best_val_metric,
    "test_{tune_metric}": test_metric,
    "model_path": best_model_path,
    "epochs_trained": {epochs},
}}

import json
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
    
    try:
        result = subprocess.run(
            ["python", script_path],
            cwd=os.path.dirname(script_path),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        working_dir = os.path.dirname(script_path)
        results_path = os.path.join(working_dir, "training_results.json")
        
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
