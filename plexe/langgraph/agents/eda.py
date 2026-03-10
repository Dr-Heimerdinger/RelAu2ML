"""
EDA (Exploratory Data Analysis) Agent.

This agent transforms relational databases into heterogeneous graphs,
performs comprehensive exploratory data analysis, and prepares data for modeling.
"""

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
from plexe.langgraph.tools.eda import (
    analyze_csv_statistics,
    detect_data_quality_issues,
    analyze_temporal_patterns,
    analyze_table_relationships,
    generate_eda_summary,
)
from plexe.langgraph.prompts.eda import EDA_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class EDAAgent(BaseAgent):
    """Agent for schema analysis, data export, and exploratory data analysis."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        tools = [
            validate_db_connection,
            export_tables_to_csv,
            extract_schema_metadata,
            analyze_csv_statistics,
            detect_data_quality_issues,
            analyze_temporal_patterns,
            analyze_table_relationships,
            generate_eda_summary,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="eda",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return EDA_SYSTEM_PROMPT
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with database and EDA-specific information."""
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
EXECUTE THESE STEPS:
1. extract_schema_metadata(db_connection_string) - analyze database schema
2. export_tables_to_csv(db_connection_string, working_dir/csv_files) - export data
3. analyze_csv_statistics(working_dir/csv_files) - get statistics  
4. detect_data_quality_issues(working_dir/csv_files) - find issues
5. analyze_temporal_patterns(working_dir/csv_files) - find timestamps
6. analyze_table_relationships(working_dir/csv_files) - classify tables
7. generate_eda_summary(working_dir/csv_files) - create report
""")
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and extract schema/CSV/EDA information."""
        base_result = super()._process_result(result, state)
        
        eda_info = {}
        working_dir = state.get("working_dir", "")
        csv_dir = f"{working_dir}/csv_files" if working_dir else None
        
        messages = result.get("messages", [])
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_result = tool_call.get("result", {})
                    
                    if tool_name == "export_tables_to_csv":
                        # csv_dir will be validated via filesystem check below
                        pass
                    
                    if tool_name == "extract_schema_metadata":
                        if isinstance(tool_result, dict) and "tables" in tool_result:
                            base_result["schema_info"] = tool_result
                    
                    if tool_name == "analyze_csv_statistics" and isinstance(tool_result, dict):
                        if tool_result.get("status") == "success":
                            eda_info["statistics"] = tool_result.get("statistics")
                    
                    if tool_name == "detect_data_quality_issues" and isinstance(tool_result, dict):
                        if tool_result.get("status") == "success":
                            eda_info["quality_issues"] = tool_result.get("quality_issues")
                    
                    if tool_name == "analyze_temporal_patterns" and isinstance(tool_result, dict):
                        if tool_result.get("status") == "success":
                            eda_info["temporal_analysis"] = tool_result.get("temporal_analysis")
                            eda_info["suggested_splits"] = tool_result.get("suggested_splits")
                    
                    if tool_name == "analyze_table_relationships" and isinstance(tool_result, dict):
                        if tool_result.get("status") == "success":
                            eda_info["relationship_analysis"] = tool_result.get("relationship_analysis")
                    
                    if tool_name == "generate_eda_summary" and isinstance(tool_result, dict):
                        if tool_result.get("status") == "success":
                            eda_info["summary"] = tool_result.get("summary")
        
        # Only set csv_dir if export actually produced files
        import os
        if csv_dir and os.path.isdir(csv_dir) and any(
            f.endswith('.csv') for f in os.listdir(csv_dir)
        ):
            base_result["csv_dir"] = csv_dir
        elif csv_dir:
            logger.warning(f"CSV directory {csv_dir} does not exist or is empty — not setting csv_dir")
            base_result["active_errors"] = [
                f"EDA completed but no CSV files were exported to {csv_dir}. "
                "Ensure the database connection string is correct and export_tables_to_csv was called."
            ]

        if eda_info:
            base_result["eda_info"] = eda_info

        base_result["current_phase"] = PipelinePhase.DATASET_BUILDING.value

        return base_result
