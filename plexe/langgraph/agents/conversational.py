"""
Conversational Agent for user interaction and requirements gathering.

This agent guides users through ML model definition via natural conversation,
validates inputs, and initiates the model building process.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

from plexe.langgraph.agents.base import BaseAgent, extract_text_content
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.conversational import get_dataset_preview
from plexe.langgraph.tools.graph_architect import validate_db_connection
from plexe.langgraph.prompts.conversational import CONVERSATIONAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class ConversationalAgent(BaseAgent):
    """Agent for conversational requirements gathering and user interaction."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        tools = [
            get_dataset_preview,
            validate_db_connection,
        ]
        
        if additional_tools:
            tools.extend(additional_tools)
        
        super().__init__(
            agent_type="conversational",
            config=config,
            tools=tools,
        )
    
    @property
    def system_prompt(self) -> str:
        return CONVERSATIONAL_SYSTEM_PROMPT
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process conversation result and detect readiness to proceed."""
        base_result = super()._process_result(result, state)
        
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if last_message:
            raw_content = last_message.content if hasattr(last_message, 'content') else ""
            content = extract_text_content(raw_content).lower()
            logger.info(f"ConversationalAgent response: {content[:200]}...")
            
            ready_indicators = [
                "ready to proceed",
                "start building",
                "begin training",
                "initiate pipeline",
                "all requirements gathered",
                "let's begin",
                "let me start",
                "i'll start",
                "proceed with",
                "starting the pipeline",
                "begin the process",
            ]
            
            if any(indicator in content for indicator in ready_indicators):
                logger.info("Detected ready indicator, setting user_confirmation_required")
                base_result["user_confirmation_required"] = True
                base_result["user_confirmed"] = True
                base_result["user_intent"] = self._extract_intent_from_state(state)
                base_result["user_confirmation_context"] = {
                    "type": "proceed_to_pipeline",
                    "message": "Ready to start the ML pipeline"
                }
        
        has_db = state.get("db_connection_string")
        has_task = any("predict" in msg.get("content", "").lower() for msg in state.get("messages", []))
        if has_db and has_task and not base_result.get("user_intent"):
            logger.info("Auto-detecting intent from state")
            base_result["user_intent"] = self._extract_intent_from_state(state)
        
        return base_result
    
    def _extract_intent_from_state(self, state: PipelineState) -> Dict[str, Any]:
        """Extract intent from state messages using keyword heuristics."""
        intent = {
            "prediction_target": None,
            "entity_type": None,
            "task_type": "binary_classification",  # conservative default
            "data_source": "database" if state.get("db_connection_string") else "csv",
            "confirmed": True,
        }

        regression_keywords = [
            "sum", "total", "sales", "revenue", "amount", "count",
            "how much", "how many", "average", "mae", "rmse", "r2",
            "mean absolute", "root mean", "ltv", "lifetime value",
            "clicks", "votes", "popularity", "number of",
        ]
        link_prediction_keywords = [
            "recommend", "which items", "list of", "purchase list",
            "map@", "precision@", "recall@", "link prediction",
        ]
        classification_keywords = [
            "churn", "leave", "cancel", "dnf", "qualify", "will happen",
            "yes or no", "probability of", "predict if", "predict whether",
        ]

        for msg in state.get("messages", []):
            content = msg.get("content", "")
            content_lower = content.lower()
            if "predict" not in content_lower and "forecast" not in content_lower:
                continue

            intent["prediction_target"] = content[:200]

            if any(kw in content_lower for kw in link_prediction_keywords):
                intent["task_type"] = "link_prediction"
            elif any(kw in content_lower for kw in regression_keywords):
                intent["task_type"] = "regression"
            elif any(kw in content_lower for kw in classification_keywords):
                intent["task_type"] = "binary_classification"
            # else: keep default "binary_classification"
            break

        return intent
