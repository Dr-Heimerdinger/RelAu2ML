"""
Relational GNN Specialist Agent.

This agent generates and executes GNN training scripts using
the plexe.relbench.modeling modules.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.gnn_specialist import generate_training_script
from plexe.langgraph.prompts.gnn_specialist import GNN_SPECIALIST_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class RelationalGNNSpecialistAgent(BaseAgent):
    """
    Agent for GNN training script generation with Training-Free HPO.
    
    This agent uses MCP (Model Context Protocol) to access external
    knowledge sources for hyperparameter optimization without training.
    MCP tools are loaded automatically via MCPManager in BaseAgent.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        # Core GNN-specific tools (non-MCP)
        # NOTE: save_artifact is intentionally excluded to prevent the LLM from
        # bypassing generate_training_script and writing broken placeholder scripts.
        tools = [
            generate_training_script,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="gnn_specialist",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return GNN_SPECIALIST_SYSTEM_PROMPT
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with training-specific information."""
        context_parts = []
        
        working_dir = state.get("working_dir", "")
        csv_dir = state.get("csv_dir", "")
        
        context_parts.append(f"Working directory: {working_dir}")
        context_parts.append(f"CSV directory: {csv_dir}")
        
        dataset_info = state.get("dataset_info")
        if dataset_info and isinstance(dataset_info, dict):
            context_parts.append(f"Dataset file: {dataset_info.get('file_path', working_dir + '/dataset.py')}")
            context_parts.append(f"Dataset class: {dataset_info.get('class_name', 'GenDataset')}")
        else:
            context_parts.append(f"Dataset file: {working_dir}/dataset.py")
            context_parts.append(f"Dataset class: GenDataset")
        
        task_info = state.get("task_info")
        if task_info and isinstance(task_info, dict):
            context_parts.append(f"Task file: {task_info.get('file_path', working_dir + '/task.py')}")
            context_parts.append(f"Task class: {task_info.get('class_name', 'GenTask')}")
            context_parts.append(f"Task type: {task_info.get('task_type', 'binary_classification')}")
            task_type = task_info.get("task_type", "binary_classification")
        else:
            context_parts.append(f"Task file: {working_dir}/task.py")
            context_parts.append(f"Task class: GenTask")
            context_parts.append(f"Task type: binary_classification")
            task_type = "binary_classification"

        user_metric = None
        user_intent = state.get("user_intent")
        if isinstance(user_intent, dict):
            user_metric = user_intent.get("evaluation_metric")
        if user_metric:
            context_parts.append(f"User-requested evaluation metric: {user_metric}")
            context_parts.append(f"YOU MUST use tune_metric based on '{user_metric}'. Do NOT override with a different metric.")
        
        # Build dataset characteristics for HPO search
        # Safely get schema_info - handle None case explicitly
        schema_info = state.get("schema_info")
        num_tables = 0
        if schema_info and isinstance(schema_info, dict):
            tables = schema_info.get("tables", {})
            if tables:
                num_tables = len(tables)
        
        dataset_chars = {
            "num_tables": num_tables,
            "num_nodes": 10000,  # Estimate - would be calculated from schema
            "is_temporal": True,  # Always true for RelBench tasks
        }
        
        context_parts.append(f"""
EXECUTE THESE STEPS (Training-Free HPO via MCP):

1. SEARCH FOR OPTIMAL HYPERPARAMETERS using MCP tools:

   a) HEURISTICS - search_optimal_hyperparameters(
       task_type="{task_type}",
       num_nodes={dataset_chars.get('num_nodes', 10000)},
       num_tables={dataset_chars.get('num_tables', 5)},
       is_temporal={dataset_chars.get('is_temporal', True)},
       model_architecture="gnn"
   )
   # Returns: Rule-based hyperparameters

   b) GOOGLE SCHOLAR - search_gnn_papers_for_hyperparameters(
       task_type="{task_type}",
       model_type="Graph Neural Network",
       limit=5
   )
   # Returns: Hyperparameters extracted from Google Scholar papers

   c) ARXIV PAPERS - search_arxiv_papers(
       query="Graph Neural Network {task_type} hyperparameters",
       max_results=5
   )
   # Returns: Recent preprints with methodology details

   d) ENSEMBLE VOTING - compare_hyperparameter_configs(
       configs=[results_from_a, results_from_b, results_from_c],
       strategy="ensemble_median"
   )
   # Returns: Final recommended hyperparameters via ensemble voting

2. GENERATE TRAINING SCRIPT with optimal hyperparameters:
   generate_training_script(
       dataset_module_path="{working_dir}/dataset.py",
       dataset_class_name="GenDataset",
       task_module_path="{working_dir}/task.py",
       task_class_name="GenTask",
       working_dir="{working_dir}",
       csv_dir="{csv_dir}",
       task_type="{task_type}",
       **recommended_hyperparameters  # Use result from step 1d
   )

3. Report the selected hyperparameters with reasoning from sources.

IMPORTANT:
- All HPO tools are provided via MCP (Model Context Protocol)
- Training execution will be handled by the Operation Agent
- Focus on selecting optimal hyperparameters WITHOUT training experiments
- ALWAYS use generate_training_script tool. NEVER write training scripts manually.
- If MCP tools are unavailable or fail, skip HPO search and call generate_training_script
  with default hyperparameters immediately. Do NOT write a placeholder script yourself.
""")
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and prepare for operation phase."""
        base_result = super()._process_result(result, state)
        
        import os
        
        working_dir = state.get("working_dir", "")
        script_path = os.path.join(working_dir, "train_script.py")
        
        if os.path.exists(script_path):
            base_result["training_script_ready"] = True
            base_result["training_script_path"] = script_path
            logger.info(f"Training script generated at {script_path}")
        else:
            error_msg = f"Training script not generated at {script_path}"
            logger.error(error_msg)
            base_result["active_errors"] = [error_msg]
            base_result["training_script_ready"] = False
        
        # Store selected hyperparameters in state for Operation Agent
        if "hyperparameters" in result:
            base_result["selected_hyperparameters"] = result["hyperparameters"]
        
        # Transition to OPERATION phase for execution
        base_result["current_phase"] = PipelinePhase.OPERATION.value
        
        return base_result
