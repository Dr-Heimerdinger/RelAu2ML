"""
Dataset Builder Agent.

This agent builds RelBench Database objects from CSV files,
generating complete Python Dataset classes for GNN training.
"""

import logging
import os
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.dataset_builder import (
    get_csv_files_info,
    get_temporal_statistics,
    register_dataset_code,
)
from plexe.langgraph.prompts.dataset_builder import DATASET_BUILDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class DatasetBuilderAgent(BaseAgent):
    """Agent for building RelBench Dataset classes from CSV data."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        tools = [
            get_csv_files_info,
            get_temporal_statistics,
            register_dataset_code,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="dataset_builder",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return DATASET_BUILDER_SYSTEM_PROMPT
    
    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        """
        Invoke the agent with retry mechanism for incomplete tasks.
        
        The agent will be re-invoked if it fails to create the dataset.py file,
        with a stronger prompt to complete the task.
        """
        working_dir = state.get("working_dir", "")
        dataset_path = os.path.join(working_dir, "dataset.py") if working_dir else ""
        max_retries = self.config.retry_config.get("dataset_building", 2)

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(f"DatasetBuilderAgent retry attempt {attempt}/{max_retries}")
                if self.emitter:
                    self.emitter.emit_thought(
                        self.name, 
                        f"Retry attempt {attempt}: Previous attempt did not create dataset.py. Retrying with stronger instructions..."
                    )
            
            # Call parent invoke
            result = super().invoke(state)
            
            # Check if dataset.py was created
            if dataset_path and os.path.exists(dataset_path):
                logger.info(f"DatasetBuilderAgent successfully created {dataset_path}")
                return result
            
            if attempt < max_retries:
                # Collect tool results from the failed attempt so the retry
                # does not have to re-call expensive tools (e.g. get_csv_files_info,
                # get_temporal_statistics on large datasets).
                tool_context = ""
                if hasattr(self, '_last_tool_results') and self._last_tool_results:
                    tool_context = "\n\nPREVIOUS TOOL RESULTS (do NOT re-call these tools):\n"
                    for tr in self._last_tool_results:
                        tool_context += f"\n--- {tr['tool_name']} ---\n{tr['content']}\n"

                retry_message = self._build_retry_message(working_dir, tool_context)

                # Update messages in state for retry
                messages = state.get("messages", [])
                messages.append({
                    "role": "user",
                    "content": retry_message,
                })
                state = {**state, "messages": messages}

        # All retries exhausted, return the last result (which has error)
        return result

    def _build_retry_message(self, working_dir: str, tool_context: str = "") -> str:
        """Build a retry message to force agent to complete the task."""
        return f"""
CRITICAL: YOU DID NOT COMPLETE YOUR TASK!

The file {working_dir}/dataset.py does NOT exist.
You MUST call register_dataset_code() to create this file.

DO NOT call get_csv_files_info or get_temporal_statistics again — their results are below.
DO NOT analyze again. DO NOT explain. Just execute these steps NOW:

1. Generate the complete GenDataset class code using the tool results below
2. Call register_dataset_code(code, "GenDataset", "{working_dir}/dataset.py")

Your response MUST include a tool call to register_dataset_code.
If you do not call this tool, you have FAILED completely.

EXECUTE THE TOOL CALL NOW.
{tool_context}
"""
    
    @staticmethod
    def _extract_db_name(connection_string: str) -> str:
        """Extract the database name from a connection string.

        E.g. 'postgresql+psycopg2://user:pass@host:5432/stack_full' → 'stack_full'
        """
        if not connection_string:
            return ""
        try:
            # Handle SQLAlchemy-style connection strings
            parts = connection_string.rsplit("/", 1)
            if len(parts) == 2:
                # Remove query params if any
                db_name = parts[1].split("?")[0]
                return db_name
        except Exception:
            pass
        return ""

    def _build_context(self, state: PipelineState) -> str:
        """Build context with CSV, schema, and EDA information."""
        context_parts = []

        working_dir = state.get('working_dir', '')
        csv_dir = state.get('csv_dir', '')
        db_name = self._extract_db_name(state.get('db_connection_string', ''))

        # Convert to absolute paths for the agent
        if working_dir:
            working_dir = os.path.abspath(working_dir)
            context_parts.append(f"Working directory: {working_dir}")

        if csv_dir:
            csv_dir = os.path.abspath(csv_dir)
            context_parts.append(f"CSV directory: {csv_dir}")
        
        if state.get("schema_info"):
            schema = state["schema_info"]
            tables = list(schema.get("tables", {}).keys())
            context_parts.append(f"Tables: {', '.join(tables)}")
            
            if schema.get("relationships"):
                rels = []
                for r in schema["relationships"]:
                    rels.append(f"{r['source_table']}.{r['source_column']} -> {r['target_table']}")
                context_parts.append(f"Foreign keys: {'; '.join(rels)}")
            
            if schema.get("temporal_columns"):
                for table, cols in schema["temporal_columns"].items():
                    context_parts.append(f"{table} time columns: {cols}")
        
        if state.get("eda_info"):
            eda = state["eda_info"]
            context_parts.append("\n## EDA Analysis Results:")
            
            if eda.get("quality_issues"):
                context_parts.append("Data Quality Issues:")
                for table, issues in eda["quality_issues"].items():
                    if issues:
                        context_parts.append(f"  - {table}: {issues}")
            
            if eda.get("temporal_analysis"):
                context_parts.append("Temporal Analysis:")
                for table, analysis in eda["temporal_analysis"].items():
                    if analysis.get("time_columns"):
                        context_parts.append(f"  - {table}: time columns = {analysis['time_columns']}")
            
            if eda.get("suggested_splits"):
                splits = eda["suggested_splits"]
                # EDA produces train_end/val_end/test_end keys:
                # train_end (70th percentile) = train/val boundary → val_timestamp
                # val_end (85th percentile) = val/test boundary → test_timestamp
                val_ts = splits.get("val_timestamp") or splits.get("train_end")
                test_ts = splits.get("test_timestamp") or splits.get("val_end")
                if val_ts:
                    context_parts.append(f"Suggested val_timestamp: {val_ts}")
                if test_ts:
                    context_parts.append(f"Suggested test_timestamp: {test_ts}")
            
            if eda.get("relationship_analysis"):
                context_parts.append("Relationship Analysis:")
                for key, info in eda["relationship_analysis"].items():
                    if isinstance(info, dict) and info.get("cardinality"):
                        context_parts.append(f"  - {key}: {info['cardinality']}")
        
        task_instruction = f"""
YOU MUST COMPLETE ALL 4 STEPS IN A SINGLE RESPONSE. You will NOT get a second chance.
If you stop before calling register_dataset_code(), the pipeline FAILS PERMANENTLY.

STEP 1: Call get_csv_files_info("{csv_dir}")
  → Returns table names, columns, row counts

STEP 2: Call get_temporal_statistics("{csv_dir}", db_name="{db_name}")
  → Returns suggested val_timestamp and test_timestamp
  (If this is a known RelBench dataset, exact author timestamps are returned)

STEP 3: Generate complete GenDataset class code
  Silently analyze the tool results from Steps 1-2 (do NOT write out a separate analysis).
  Then immediately generate a complete GenDataset class with:
  - Class definition extending Dataset
  - val_timestamp = pd.Timestamp("YYYY-MM-DD") using value from Step 2
  - test_timestamp = pd.Timestamp("YYYY-MM-DD") using value from Step 2
  - __init__ method accepting csv_dir and cache_dir parameters
  - make_db() method that:
    * Loads all CSV files
    * Handles data cleaning (\\N → NaN, timezone stripping, type conversions)
    * Creates Table objects with correct fkey_col_to_pkey_table mappings
    * Sets time_col for temporal tables (or None for static tables)
    * Returns Database with all tables

STEP 4: Call register_dataset_code(code, "GenDataset", "{working_dir}/dataset.py")
  → MUST return {{"status": "registered", ...}}

RULES:
- Execute ALL tool calls. Do NOT describe what you will do — just DO it.
- Do NOT write a separate analysis section. Go directly from tool results to code generation.
- The file {working_dir}/dataset.py MUST exist when you finish.
- Your task is INCOMPLETE and FAILED unless register_dataset_code() was called successfully.
"""
        context_parts.append(task_instruction)
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and extract dataset information."""
        base_result = super()._process_result(result, state)
        
        dataset_info = {}
        working_dir = state.get("working_dir", "")
        
        # Check if dataset.py was created
        dataset_path = os.path.join(working_dir, "dataset.py")
        if os.path.exists(dataset_path):
            dataset_info["class_name"] = "GenDataset"
            dataset_info["file_path"] = dataset_path
            logger.info(f"Dataset file created at: {dataset_path}")

            # Extract val/test timestamps from generated dataset.py
            try:
                import ast as _ast
                with open(dataset_path) as f:
                    tree = _ast.parse(f.read())
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.Assign):
                        for target in node.targets:
                            if isinstance(target, _ast.Name):
                                if target.id == 'val_timestamp' and isinstance(node.value, _ast.Call):
                                    if node.value.args and isinstance(node.value.args[0], _ast.Constant):
                                        dataset_info["val_timestamp"] = node.value.args[0].value
                                elif target.id == 'test_timestamp' and isinstance(node.value, _ast.Call):
                                    if node.value.args and isinstance(node.value.args[0], _ast.Constant):
                                        dataset_info["test_timestamp"] = node.value.args[0].value
            except Exception:
                pass  # non-critical fallback; task builder validates timestamps directly

            # Warn if timestamps are None — this will crash downstream
            if not dataset_info.get("val_timestamp") or not dataset_info.get("test_timestamp"):
                logger.warning(
                    "Dataset generated with None timestamps — this will crash task initialization. "
                    "The dataset builder agent should always set val_timestamp and test_timestamp."
                )

            base_result["dataset_info"] = dataset_info
            base_result["current_phase"] = PipelinePhase.TASK_BUILDING.value

            # Cache CSV files info in state so downstream agents (TaskBuilder)
            # don't have to re-call get_csv_files_info on large datasets.
            csv_dir = state.get("csv_dir", "")
            if csv_dir:
                try:
                    csv_info = get_csv_files_info.invoke({"csv_dir": csv_dir})
                    base_result["csv_files_info"] = csv_info
                except Exception:
                    pass  # non-critical; TaskBuilder can still call the tool
        else:
            error_msg = f"CRITICAL ERROR: Dataset file not found at {dataset_path}. DatasetBuilderAgent did not complete its task. The agent must call register_dataset_code() to generate dataset.py."
            logger.error(error_msg)
            base_result["active_errors"] = [error_msg]
            base_result["error_history"] = [error_msg]
            # Set dataset_info with error flag for debugging
            dataset_info["class_name"] = "GenDataset"
            dataset_info["file_path"] = dataset_path
            dataset_info["error"] = error_msg
            base_result["dataset_info"] = dataset_info
            # Do NOT advance phase - stay in dataset building
            base_result["current_phase"] = PipelinePhase.DATASET_BUILDING.value
        
        return base_result
