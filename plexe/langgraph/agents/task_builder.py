"""
Task Builder Agent.

This agent builds RelBench Task objects for prediction tasks,
generating SQL queries and complete Python Task classes.
"""

import logging
import os
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.common import save_artifact
from plexe.langgraph.tools.dataset_builder import get_csv_files_info
from plexe.langgraph.tools.task_builder import test_sql_query, register_task_code, validate_dataset_timestamps
from plexe.langgraph.prompts.task_builder import TASK_BUILDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class TaskBuilderAgent(BaseAgent):
    """Agent for building RelBench Task classes."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        tools = [
            get_csv_files_info,
            validate_dataset_timestamps,
            test_sql_query,
            register_task_code,
            save_artifact,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="task_builder",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return TASK_BUILDER_SYSTEM_PROMPT
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with task-specific information."""
        context_parts = []
        
        if state.get("working_dir"):
            context_parts.append(f"Working directory: {state['working_dir']}")
        
        if state.get("csv_dir"):
            context_parts.append(f"CSV directory: {state['csv_dir']}")
        
        # User intent analysis
        if state.get("user_intent"):
            intent = state["user_intent"]
            context_parts.append("\n## User Intent:")
            if isinstance(intent, dict):
                pred_target = intent.get('prediction_target', 'unknown')
                task_type = intent.get('task_type', 'unknown')
                context_parts.append(f"  - Prediction target: {pred_target}")
                context_parts.append(f"  - Task type: {task_type}")
                
                # Suggest appropriate metrics
                if 'binary' in str(task_type).lower() or 'classification' in str(task_type).lower():
                    context_parts.append(f"  - Suggested metrics: average_precision, accuracy, f1, roc_auc")
                elif 'regression' in str(task_type).lower():
                    context_parts.append(f"  - Suggested metrics: mae, rmse, r2")
                elif 'link' in str(task_type).lower() or 'recommendation' in str(task_type).lower():
                    context_parts.append(f"  - Suggested metrics: link_prediction_map, link_prediction_precision, link_prediction_recall")
                    context_parts.append(f"  - Use RecommendationTask base class with eval_k parameter")
            else:
                context_parts.append(f"  - Intent: {intent}")
        
        # Schema information
        if state.get("schema_info"):
            schema = state["schema_info"]
            context_parts.append("\n## Schema Information:")
            tables = list(schema.get("tables", {}).keys())
            context_parts.append(f"Available tables: {', '.join(tables)}")
            
            context_parts.append("\nTable Details:")
            for table_name, table_info in schema.get("tables", {}).items():
                columns = table_info.get("columns", [])
                pk = table_info.get("primary_key", [])
                context_parts.append(f"  - {table_name}:")
                context_parts.append(f"    * Columns: {', '.join([c['name'] for c in columns[:10]])}")
                if pk:
                    context_parts.append(f"    * Primary Key: {pk}")
            
            # Foreign key relationships
            if schema.get("relationships"):
                context_parts.append("\nForeign Key Relationships:")
                for rel in schema["relationships"]:
                    context_parts.append(
                        f"  - {rel['source_table']}.{rel['source_column']} -> {rel['target_table']}.{rel['target_column']}"
                    )
        
        # Dataset information
        if state.get("dataset_info"):
            ds = state["dataset_info"]
            context_parts.append("\n## Dataset Information:")
            context_parts.append(f"  - Class: {ds.get('class_name')}")
            if ds.get("val_timestamp"):
                context_parts.append(f"  - Validation timestamp: {ds.get('val_timestamp')}")
            if ds.get("test_timestamp"):
                context_parts.append(f"  - Test timestamp: {ds.get('test_timestamp')}")
        
        # EDA insights
        if state.get("eda_info"):
            eda = state["eda_info"]
            context_parts.append("\n## EDA Analysis:")
            
            if eda.get("statistics"):
                context_parts.append("Table Statistics:")
                for table, stats in eda["statistics"].items():
                    if isinstance(stats, dict):
                        row_count = stats.get("row_count", "unknown")
                        context_parts.append(f"  - {table}: {row_count} rows")
            
            if eda.get("temporal_analysis"):
                context_parts.append("\nTemporal Analysis:")
                for table, analysis in eda["temporal_analysis"].items():
                    if analysis.get("time_columns"):
                        cols = analysis['time_columns']
                        context_parts.append(f"  - {table} time columns: {cols}")
                        # Add time range info if available
                        for col_name, col_info in cols.items():
                            if isinstance(col_info, dict):
                                min_date = col_info.get('min')
                                max_date = col_info.get('max')
                                if min_date and max_date:
                                    context_parts.append(f"    * {col_name}: {min_date} to {max_date}")
            
            # Suggest timedelta based on temporal data
            if eda.get("suggested_timedelta"):
                context_parts.append(f"\nSuggested prediction window: {eda.get('suggested_timedelta')}")
        
        # Task generation instructions
        working_dir = state.get('working_dir', '')
        csv_dir = state.get('csv_dir', '')
        
        context_parts.append(f"""
## Your Task:
1. Determine if this is an EntityTask or RecommendationTask based on user intent
2. Identify the entity table and entity column (or src/dst for recommendations)
3. Determine appropriate time_col (from temporal analysis)
4. Design SQL query with proper temporal filtering to compute target labels
5. Choose appropriate metrics based on task type
6. Estimate reasonable timedelta (prediction window) based on temporal data range
7. Set num_eval_timestamps (default 20, adjust based on data frequency)
8. For link prediction: set eval_k (typical: 10-12)
9. Test your SQL: test_sql_query("{csv_dir}", query)
10. Generate complete code and save: register_task_code(code, "GenTask", "{working_dir}/task.py", task_type)

CRITICAL REMINDERS:
- Use TaskType enum: TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION, TaskType.LINK_PREDICTION
- Import correct base class: EntityTask or RecommendationTask
- Import only metrics you use from plexe.relbench.metrics
- Convert timestamps: timestamp_df = pd.DataFrame({{"timestamp": timestamps}})
- Use duckdb.sql() method, not conn.execute()
- Return Table object with proper fkey_col_to_pkey_table mapping
""")
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and extract task information."""
        base_result = super()._process_result(result, state)
        
        task_info = {}
        generated_code = state.get("generated_code", {})
        working_dir = state.get("working_dir", "")
        
        task_path = os.path.join(working_dir, "task.py")
        if os.path.exists(task_path):
            task_info["class_name"] = "GenTask"
            task_info["file_path"] = task_path
            logger.info(f"Task file created at: {task_path}")
            
            intent = state.get("user_intent", {})
            if isinstance(intent, dict):
                task_info["task_type"] = intent.get("task_type", "binary_classification")
            else:
                task_info["task_type"] = "binary_classification"
        else:
            error_msg = f"CRITICAL ERROR: Task file not found at {task_path}. TaskBuilderAgent did not complete its task. The agent must call register_task_code() to generate task.py."
            logger.error(error_msg)
            # Return error state to force re-invocation or escalation
            base_result["error"] = error_msg
            base_result["status"] = "error"
            task_info["class_name"] = "GenTask"
            task_info["file_path"] = task_path
            task_info["error"] = "File not generated"
            task_info["task_type"] = "binary_classification"
        
        base_result["task_info"] = task_info
        
        if generated_code:
            base_result["generated_code"] = generated_code
        
        base_result["current_phase"] = PipelinePhase.GNN_TRAINING.value
        
        return base_result
