"""
Operation Agent.

This agent handles environment setup, execution monitoring,
and final model packaging.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.gnn_specialist import execute_training_script
from plexe.langgraph.prompts.operation import OPERATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OperationAgent(BaseAgent):
    """Agent for environment setup and execution monitoring."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        tools = [
            execute_training_script,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="operation",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return OPERATION_SYSTEM_PROMPT
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with operation-specific information."""
        context_parts = []
        
        working_dir = state.get("working_dir", "")
        context_parts.append(f"Working directory: {working_dir}")
        
        # Check if training script is ready
        training_script_ready = state.get("training_script_ready", False)
        training_script_path = state.get("training_script_path", f"{working_dir}/train_script.py")
        
        if training_script_ready:
            context_parts.append(f"Training script ready: {training_script_path}")
        
        # Check if training has been executed
        if state.get("training_result"):
            result = state["training_result"]
            context_parts.append(f"Training already completed:")
            context_parts.append(f"  - Metrics: {result.get('metrics')}")
            context_parts.append(f"  - Model path: {result.get('model_path')}")
        
        # Check for hyperparameters from GNN Specialist
        if state.get("selected_hyperparameters"):
            hp = state["selected_hyperparameters"]
            context_parts.append(f"Selected hyperparameters: {hp}")
        
        if state.get("error_history"):
            context_parts.append(f"Previous errors: {state['error_history']}")
        
        # Instructions based on state
        if not state.get("training_result"):
            context_parts.append(f"""
EXECUTE TRAINING:
1. execute_training_script(
    script_path="{training_script_path}",
    timeout=3600  # 1 hour timeout
)
2. Process the training results from {working_dir}/training_results.json
3. Report metrics and model location
""")
        else:
            context_parts.append(f"""
FINALIZE PIPELINE:
1. Review training results
2. List all generated artifacts:
   - {working_dir}/dataset.py - Dataset class
   - {working_dir}/task.py - Task class  
   - {working_dir}/train_script.py - Training script
   - {working_dir}/best_model.pt - Trained model
   - {working_dir}/training_results.json - Training metrics
3. Provide summary and deployment recommendations
""")
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and finalize the pipeline."""
        base_result = super()._process_result(result, state)
        
        import os
        import json
        
        working_dir = state.get("working_dir", ".")
        results_path = os.path.join(working_dir, "training_results.json")
        
        # Check if training was executed and results are available
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    training_results = json.load(f)
                
                base_result["training_result"] = {
                    "metrics": training_results,
                    "model_path": training_results.get("model_path"),
                    "script_path": os.path.join(working_dir, "train_script.py"),
                }
                logger.info(f"Training results processed: {training_results}")
            except Exception as e:
                logger.warning(f"Could not read training results: {e}")
                base_result["active_errors"] = [f"Failed to read training results: {e}"]
        
        # Mark pipeline as completed
        base_result["current_phase"] = PipelinePhase.COMPLETED.value
        
        return base_result
    
    def generate_inference_code(self, state: PipelineState) -> str:
        """Generate inference code for the trained model."""
        working_dir = state.get("working_dir", ".")
        # Safely get task_type - handle None case explicitly
        task_info = state.get("task_info")
        task_type = "regression"
        if task_info and isinstance(task_info, dict):
            task_type = task_info.get("task_type", "regression")
        
        inference_code = f'''"""
Auto-generated inference code for the trained GNN model.
"""

import torch
import sys
import os

sys.path.insert(0, "{working_dir}")

from dataset import GenDataset
from task import GenTask

def load_model(model_path: str):
    """Load the trained model."""
    from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
    from plexe.relbench.modeling.graph import make_pkey_fkey_graph
    from plexe.relbench.modeling.utils import get_stype_proposal
    
    # Initialize dataset and task
    csv_dir = "{working_dir}/csv_files"
    dataset = GenDataset(csv_dir=csv_dir)
    task = GenTask(dataset)
    db = dataset.get_db()
    
    # Build graph
    col_to_stype_dict = get_stype_proposal(db)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir="{working_dir}/cache/",
    )
    
    # Recreate model architecture
    class GNNModel(torch.nn.Module):
        def __init__(self, data, col_stats_dict, hidden_channels=128, out_channels=1):
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
                num_layers=2,
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
    
    model = GNNModel(data, col_stats_dict)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, data, task


def predict(model, data, task, entities, timestamp):
    """
    Make predictions for given entities at a specific timestamp.
    
    Args:
        model: Trained GNN model
        data: HeteroData graph
        task: Task definition
        entities: List of entity IDs to predict for
        timestamp: Prediction timestamp
    
    Returns:
        Predictions for each entity
    """
    from plexe.relbench.modeling.graph import get_node_train_table_input
    from torch_geometric.loader import NeighborLoader
    import pandas as pd
    
    # Create prediction table
    pred_df = pd.DataFrame({{
        task.time_col: [pd.Timestamp(timestamp)] * len(entities),
        task.entity_col: entities,
    }})
    
    # Create loader
    # ... (implementation depends on specific use case)
    
    return predictions


if __name__ == "__main__":
    model_path = "{working_dir}/best_model.pt"
    model, data, task = load_model(model_path)
    print("Model loaded successfully!")
'''
        
        return inference_code
