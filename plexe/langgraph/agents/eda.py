"""EDA (Exploratory Data Analysis) Agent."""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.graph_architect import (
    validate_db_connection,
    export_tables_to_csv,
    extract_schema_metadata,
)
from plexe.langgraph.tools.eda import analyze_all_csv
from plexe.langgraph.prompts.eda import EDA_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class EDAAgent(BaseAgent):
    """Agent for schema analysis, data export, and exploratory data analysis."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
        token_tracker=None,
    ):
        tools = [
            validate_db_connection,
            export_tables_to_csv,
            extract_schema_metadata,
            analyze_all_csv,
        ]
        if additional_tools:
            tools.extend(additional_tools)

        super().__init__(
            agent_type="eda",
            config=config,
            tools=tools,
            token_tracker=token_tracker,
        )

    @property
    def system_prompt(self) -> str:
        return EDA_SYSTEM_PROMPT

    def _build_context(self, state: PipelineState) -> str:
        context_parts = []

        if state.get("working_dir"):
            context_parts.append(f"Working directory: {state['working_dir']}")
            context_parts.append(f"CSV output directory: {state['working_dir']}/csv_files")

        if state.get("db_connection_string"):
            context_parts.append(f"Database: {state['db_connection_string']}")

        if state.get("user_intent"):
            intent = state["user_intent"]
            if isinstance(intent, dict):
                context_parts.append(f"Prediction target: {intent.get('prediction_target', 'unknown')}")
            else:
                context_parts.append(f"User intent: {intent}")

        context_parts.append("""
EXECUTE THESE STEPS IN ORDER:
1. extract_schema_metadata(db_connection_string)
2. export_tables_to_csv(db_connection_string, working_dir/csv_files)
3. analyze_all_csv(working_dir/csv_files, schema_info_from_step_1)
   Pass the schema result from step 1 as schema_info so table classification works.
""")

        return "\n".join(context_parts)

    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        import json
        import os
        from langchain_core.messages import ToolMessage

        base_result = super()._process_result(result, state)

        eda_info = {}
        working_dir = state.get("working_dir", "")
        csv_dir = os.path.abspath(os.path.join(working_dir, "csv_files")) if working_dir else None

        # Extract tool results from ToolMessage objects (not AIMessage.tool_calls)
        messages = result.get("messages", [])
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            try:
                content = msg.content
                tool_result = json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                tool_result = {}
            if not isinstance(tool_result, dict):
                continue

            if msg.name == "extract_schema_metadata":
                if "tables" in tool_result:
                    base_result["schema_info"] = tool_result

            elif msg.name == "analyze_all_csv":
                if tool_result.get("status") == "success":
                    eda_info["statistics"] = tool_result.get("statistics")
                    eda_info["quality_issues"] = tool_result.get("quality_issues")
                    eda_info["temporal_analysis"] = tool_result.get("temporal_analysis")
                    eda_info["suggested_splits"] = tool_result.get("suggested_splits")
                    eda_info["relationship_analysis"] = tool_result.get("relationship_analysis")
                    eda_info["summary"] = tool_result.get("summary")

        # Filesystem fallback for csv_dir
        if csv_dir and os.path.isdir(csv_dir) and any(
            f.endswith('.csv') for f in os.listdir(csv_dir)
        ):
            base_result["csv_dir"] = csv_dir
        elif csv_dir:
            logger.warning(f"CSV directory {csv_dir} does not exist or is empty")
            base_result["active_errors"] = [
                f"EDA completed but no CSV files were exported to {csv_dir}. "
                "Ensure the database connection string is correct and export_tables_to_csv was called."
            ]

        if eda_info:
            base_result["eda_info"] = eda_info

        base_result["current_phase"] = PipelinePhase.DATASET_BUILDING.value
        return base_result
