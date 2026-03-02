"""
Task Builder Agent.

This agent builds RelBench Task objects for prediction tasks,
generating SQL queries and complete Python Task classes.
"""

import logging
import os
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.common import save_artifact
from plexe.langgraph.tools.dataset_builder import get_csv_files_info
from plexe.langgraph.tools.task_builder import test_sql_query, register_task_code, validate_dataset_timestamps, fix_dataset_timestamps, analyze_task_structure, determine_lookback_window
from plexe.langgraph.prompts.task_builder import TASK_BUILDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Maximum retry attempts for incomplete tasks
MAX_AGENT_RETRIES = 2

class TaskBuilderAgent(BaseAgent):
    """Agent for building RelBench Task classes."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        tools = [
            get_csv_files_info,
            analyze_task_structure,
            determine_lookback_window,
            validate_dataset_timestamps,
            fix_dataset_timestamps,
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
    
    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        """
        Invoke the agent with retry mechanism for incomplete tasks.
        
        The agent will be re-invoked if it fails to create the task.py file,
        with a stronger prompt to complete the task.
        """
        working_dir = state.get("working_dir", "")
        task_path = os.path.join(working_dir, "task.py") if working_dir else ""
        
        for attempt in range(MAX_AGENT_RETRIES + 1):
            if attempt > 0:
                logger.warning(f"TaskBuilderAgent retry attempt {attempt}/{MAX_AGENT_RETRIES}")
                if self.emitter:
                    self.emitter.emit_thought(
                        self.name, 
                        f"Retry attempt {attempt}: Previous attempt did not create task.py. Retrying with stronger instructions..."
                    )
            
            # Call parent invoke
            result = super().invoke(state)
            
            # Check if task.py was created
            if task_path and os.path.exists(task_path):
                logger.info(f"TaskBuilderAgent successfully created {task_path}")
                return result
            
            # If this is not the last attempt, prepare for retry
            if attempt < MAX_AGENT_RETRIES:
                # Add a follow-up message to force completion
                retry_message = self._build_retry_message(working_dir, state.get("csv_dir", ""))
                
                # Update messages in state for retry
                messages = state.get("messages", [])
                messages.append({
                    "role": "user",
                    "content": retry_message,
                })
                state = {**state, "messages": messages}
        
        # All retries exhausted, return the last result (which has error)
        return result
    
    def _build_retry_message(self, working_dir: str, csv_dir: str) -> str:
        """Build a retry message to force agent to complete the task."""
        return f"""
CRITICAL: YOU DID NOT COMPLETE YOUR TASK!

The file {working_dir}/task.py does NOT exist.
You MUST call register_task_code() to create this file.

DO NOT analyze again. DO NOT explain. Just execute these steps NOW:

1. Generate the complete GenTask class code with SQL query
2. Call register_task_code(code, "GenTask", "{working_dir}/task.py", task_type)

Your response MUST include a tool call to register_task_code.
If you do not call this tool, you have FAILED completely.

EXECUTE THE TOOL CALL NOW.
"""
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with task-specific information."""
        context_parts = []
        
        working_dir = state.get("working_dir", "")
        csv_dir = state.get("csv_dir", "")
        
        if working_dir:
            context_parts.append(f"Working directory: {working_dir}")
        
        if csv_dir:
            context_parts.append(f"CSV directory: {csv_dir}")
        
        # Pre-check: Verify dataset.py exists
        dataset_path = os.path.join(working_dir, "dataset.py") if working_dir else ""
        if dataset_path and not os.path.exists(dataset_path):
            context_parts.append(f"""
## CRITICAL ERROR - PREREQUISITE NOT MET
Dataset file does not exist at: {dataset_path}
The DatasetBuilderAgent must complete successfully before TaskBuilderAgent can run.
You CANNOT proceed without dataset.py. Report this error.
""")
        
        # User intent analysis
        if state.get("user_intent"):
            intent = state["user_intent"]
            context_parts.append("\n## User Intent:")
            if isinstance(intent, dict):
                pred_target = intent.get('prediction_target', 'unknown')
                task_type = intent.get('task_type', 'unknown')
                eval_metric = intent.get('evaluation_metric')
                context_parts.append(f"  - Prediction target: {pred_target}")
                context_parts.append(f"  - Task type: {task_type}")
                if eval_metric:
                    context_parts.append(f"  - User's evaluation metric: {eval_metric}")

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
                context_parts.append(f"    * Columns: {', '.join([c['name'] for c in columns[:20]])}")
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
        
        # Compute the val/test gap from dataset_info so the agent knows
        # the maximum timedelta the dataset can support.
        # Fall back to EDA suggested_splits when dataset_info has no timestamps.
        ds = state.get("dataset_info") or {}
        eda = state.get("eda_info") or {}
        splits = eda.get("suggested_splits") or {}

        val_ts_str = ds.get("val_timestamp") or splits.get("train_end", "")
        test_ts_str = ds.get("test_timestamp") or splits.get("val_end", "")
        timestamp_gap_days = None
        if val_ts_str and test_ts_str:
            try:
                import pandas as _pd
                _gap = (_pd.Timestamp(test_ts_str) - _pd.Timestamp(val_ts_str)).days
                timestamp_gap_days = int(_gap)
            except Exception:
                pass

        # Task generation instructions
        working_dir = state.get('working_dir', '')
        csv_dir = state.get('csv_dir', '')
        dataset_file = f"{working_dir}/dataset.py"

        context_parts.append(f"""
## Your Task (follow the Mandatory Workflow in the system prompt):
1. Determine task type from user intent and metric
2. Validate dataset timestamps: call validate_dataset_timestamps("{dataset_file}", "{csv_dir}", timedelta_days)
   where timedelta_days is the prediction window you plan to use.{f'''
   NOTE: Current val/test gap is {timestamp_gap_days} days. Your timedelta MUST be <= {timestamp_gap_days}.''' if timestamp_gap_days else ''}
   If invalid, fix with fix_dataset_timestamps() before proceeding.
3. Choose base class (EntityTask or RecommendationTask)
4. Call analyze_task_structure() to get evidence for pattern selection:
   analyze_task_structure("{csv_dir}", event_table, entity_col, time_col, timedelta_days, task_description, entity_table)
   Review ALL sections of the output, especially:
   - entity_source.entity_table_has_creation_date (Pattern D signal)
   - temporal.max_gap_exceeds_timedelta (if true, prefer Pattern B over A even with an entity table)
   - temporal.suggested_lookback_interval (use this for Pattern B lookback value)
   - pattern_candidates (ranked suggestions -- use as guidance, not as absolute rule)
   - building_blocks (whether CTE, nested JOIN, quality filter, or HAVING is needed)
   - schema_hints.potential_join_tables (tables that share columns with the event table)
5. Identify entity table, entity column, time column, and target column
6. Design SQL query using your selected pattern. Consider composable building blocks (Part 4B) if needed:
   - CTE: for multi-table preprocessing or combining event sources
   - Nested LEFT JOIN: for entity-event pre-join patterns
   - Chained JOIN: for link prediction through junction tables
   - Quality filters: for rating/length/status conditions
7. Choose appropriate metrics based on task type
8. For link prediction: set eval_k (typical: 10-12)
9. Test your SQL: test_sql_query("{csv_dir}", query)
10. Generate complete code and save: register_task_code(code, "GenTask", "{working_dir}/task.py", task_type)

## File Paths:
- Dataset file: {dataset_file}
- CSV directory: {csv_dir}
- Task output: {working_dir}/task.py

## Reminders:
- Use TaskType enum: TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION, TaskType.LINK_PREDICTION
- "predict if", "whether", "will make any", "will X do more than N" = BINARY_CLASSIFICATION, NOT REGRESSION (use IF(COUNT>=1,1,0) not raw COUNT)
- Import correct base class: EntityTask or RecommendationTask
- Import only metrics you use from plexe.relbench.metrics
- Column names in SQL must EXACTLY match CSV column names INCLUDING CASE (use get_csv_files_info to verify)
- entity_table must EXACTLY match the CSV filename (without .csv) INCLUDING CASE
- time_col value must match the timestamp column name in the SQL output
- ALWAYS use INTERVAL '{{self.timedelta}}' in SQL — NEVER compute days manually
- Do NOT add CAST(col AS TIMESTAMP) for columns already parsed as datetime in dataset.py
- Use duckdb.register() for every DataFrame, then duckdb.sql()
- Return Table with proper fkey_col_to_pkey_table mapping
- If data range is narrow (< 3 months) or data_range/timedelta < 5, set num_eval_timestamps = 40
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
            
            base_result["task_info"] = task_info
            base_result["current_phase"] = PipelinePhase.GNN_TRAINING.value
        else:
            error_msg = f"CRITICAL ERROR: Task file not found at {task_path}. TaskBuilderAgent did not complete its task. The agent must call register_task_code() to generate task.py."
            logger.error(error_msg)
            # Add to errors list so routing detects failure
            existing_errors = base_result.get("errors", []) or []
            existing_errors.append(error_msg)
            base_result["errors"] = existing_errors
            base_result["status"] = "error"
            # Set task_info with error flag for debugging
            task_info["class_name"] = "GenTask"
            task_info["file_path"] = task_path
            task_info["error"] = error_msg
            task_info["task_type"] = "binary_classification"
            base_result["task_info"] = task_info
            # Do NOT advance phase - stay in task building
            base_result["current_phase"] = PipelinePhase.TASK_BUILDING.value
        
        if generated_code:
            base_result["generated_code"] = generated_code
        
        return base_result
