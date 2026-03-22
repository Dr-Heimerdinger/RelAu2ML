"""Dataset Builder Agent."""

import logging
import os
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

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
        token_tracker=None,
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
            token_tracker=token_tracker,
        )

    @property
    def system_prompt(self) -> str:
        return DATASET_BUILDER_SYSTEM_PROMPT

    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        working_dir = state.get("working_dir", "")
        dataset_path = os.path.join(working_dir, "dataset.py") if working_dir else ""
        max_retries = self.config.retry_config.get("dataset_building", 2)

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(f"DatasetBuilderAgent retry attempt {attempt}/{max_retries}")
                if self.emitter:
                    self.emitter.emit_thought(
                        self.name,
                        f"Retry attempt {attempt}: Previous attempt did not create dataset.py.",
                    )

            result = super().invoke(state)

            if dataset_path and os.path.exists(dataset_path):
                logger.info(f"DatasetBuilderAgent successfully created {dataset_path}")
                return result

            if attempt < max_retries:
                tool_context = ""
                if hasattr(self, '_last_tool_results') and self._last_tool_results:
                    tool_context = "\n\nPREVIOUS TOOL RESULTS (do NOT re-call these tools):\n"
                    for tr in self._last_tool_results:
                        tool_context += f"\n--- {tr['tool_name']} ---\n{tr['content']}\n"

                retry_message = self._build_retry_message(working_dir, tool_context)
                messages = state.get("messages", [])
                messages.append({"role": "user", "content": retry_message})
                state = {**state, "messages": messages}

        return result

    def _build_retry_message(self, working_dir: str, tool_context: str = "") -> str:
        return f"""CRITICAL: dataset.py does NOT exist at {working_dir}/dataset.py.
You MUST call register_dataset_code() to create it.
Do NOT re-call get_csv_files_info or get_temporal_statistics.

1. Generate the complete GenDataset class code using the tool results below.
2. Call register_dataset_code(code, "GenDataset", "{working_dir}/dataset.py")
{tool_context}"""

    @staticmethod
    def _extract_db_name(connection_string: str) -> str:
        if not connection_string:
            return ""
        try:
            parts = connection_string.rsplit("/", 1)
            if len(parts) == 2:
                return parts[1].split("?")[0]
        except Exception:
            pass
        return ""

    def _build_context(self, state: PipelineState) -> str:
        context_parts = []

        working_dir = state.get('working_dir', '')
        csv_dir = state.get('csv_dir', '')
        db_name = self._extract_db_name(state.get('db_connection_string', ''))

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
            for tname, tinfo in schema.get("tables", {}).items():
                cols = [c["name"] for c in tinfo.get("columns", [])]
                if cols:
                    context_parts.append(f"  {tname} columns: {cols}")
            if schema.get("relationships"):
                rels = [f"{r['source_table']}.{r['source_column']} -> {r['target_table']}" for r in schema["relationships"]]
                context_parts.append(f"Foreign keys: {'; '.join(rels)}")
            if schema.get("temporal_columns"):
                for table, cols in schema["temporal_columns"].items():
                    context_parts.append(f"{table} time columns: {cols}")

        # Use EDA results to provide temporal info (avoid re-scanning)
        eda = state.get("eda_info") or {}
        if eda.get("quality_issues"):
            context_parts.append("\n## Data Quality Issues:")
            for table, issues in eda["quality_issues"].items():
                if isinstance(issues, dict) and issues.get("issues"):
                    context_parts.append(f"  - {table}: {len(issues['issues'])} issues")

        if eda.get("temporal_analysis"):
            context_parts.append("\n## Temporal Columns (from EDA):")
            for table, analysis in eda["temporal_analysis"].items():
                if isinstance(analysis, dict) and analysis.get("temporal_columns"):
                    for col, info in analysis["temporal_columns"].items():
                        context_parts.append(f"  - {table}.{col}: {info.get('min')} to {info.get('max')}")

        if eda.get("suggested_splits"):
            splits = eda["suggested_splits"]
            val_ts = splits.get("val_timestamp") or splits.get("train_end")
            test_ts = splits.get("test_timestamp") or splits.get("val_end")
            if val_ts:
                context_parts.append(f"EDA suggested val_timestamp: {val_ts}")
            if test_ts:
                context_parts.append(f"EDA suggested test_timestamp: {test_ts}")

        if eda.get("relationship_analysis"):
            rel = eda["relationship_analysis"]
            dim_fact = rel.get("dimension_fact_classification", {})
            if dim_fact:
                context_parts.append("\n## Table Classification (from EDA):")
                for tbl, info in dim_fact.items():
                    if isinstance(info, dict):
                        context_parts.append(f"  - {tbl}: {info.get('classification', 'unknown')}")

        context_parts.append(f"""
YOU MUST COMPLETE ALL STEPS IN A SINGLE RESPONSE.

STEP 1: Call get_csv_files_info("{csv_dir}")

STEP 2: Call get_temporal_statistics("{csv_dir}", db_name="{db_name}")
  Returns suggested val_timestamp and test_timestamp.
  If EDA already provided timestamps above, verify or use them directly.

STEP 3: Generate complete GenDataset class code using results from Steps 1-2.

STEP 4: Call register_dataset_code(code, "GenDataset", "{working_dir}/dataset.py")
  MUST return {{"status": "registered"}}.

The file {working_dir}/dataset.py MUST exist when you finish.
""")

        return "\n".join(context_parts)

    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        base_result = super()._process_result(result, state)

        dataset_info = {}
        working_dir = state.get("working_dir", "")

        dataset_path = os.path.join(working_dir, "dataset.py")
        if os.path.exists(dataset_path):
            dataset_info["class_name"] = "GenDataset"
            dataset_info["file_path"] = dataset_path
            logger.info(f"Dataset file created at: {dataset_path}")

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
                pass

            if not dataset_info.get("val_timestamp") or not dataset_info.get("test_timestamp"):
                logger.warning("Dataset generated with None timestamps")

            base_result["dataset_info"] = dataset_info
            base_result["current_phase"] = PipelinePhase.TASK_BUILDING.value

            # Cache CSV files info for downstream agents
            csv_dir = state.get("csv_dir", "")
            if csv_dir:
                try:
                    csv_info = get_csv_files_info.invoke({"csv_dir": csv_dir})
                    base_result["csv_files_info"] = csv_info
                except Exception:
                    pass
        else:
            error_msg = f"Dataset file not found at {dataset_path}. Agent must call register_dataset_code()."
            logger.error(error_msg)
            base_result["active_errors"] = [error_msg]
            base_result["error_history"] = [error_msg]
            dataset_info["class_name"] = "GenDataset"
            dataset_info["file_path"] = dataset_path
            dataset_info["error"] = error_msg
            base_result["dataset_info"] = dataset_info
            base_result["current_phase"] = PipelinePhase.DATASET_BUILDING.value

        return base_result
