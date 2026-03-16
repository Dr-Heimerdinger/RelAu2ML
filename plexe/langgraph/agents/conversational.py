"""
Conversational Agent for user interaction and requirements gathering.

This agent guides users through ML model definition via natural conversation,
validates inputs, and initiates the model building process.
"""

import logging
import re
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
        token_tracker=None,
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
            token_tracker=token_tracker,
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
        
        # Extract connection string from user messages if not already in state
        if not state.get("db_connection_string"):
            extracted_conn = self._extract_connection_string(state)
            if extracted_conn:
                logger.info(f"Extracted connection string from user message: {extracted_conn[:40]}...")
                base_result["db_connection_string"] = extracted_conn

        has_db = state.get("db_connection_string") or base_result.get("db_connection_string")
        has_task = any("predict" in msg.get("content", "").lower() for msg in state.get("messages", []))
        if has_db and has_task and not base_result.get("user_intent"):
            logger.info("Auto-detecting intent from state")
            base_result["user_intent"] = self._extract_intent_from_state(state)

        return base_result

    def _extract_connection_string(self, state: PipelineState) -> Optional[str]:
        """Extract a database connection string from user messages."""
        # Matches postgresql://, postgresql+psycopg2://, mysql://, sqlite:///,
        # mysql+pymysql://, etc.
        conn_pattern = re.compile(
            r'(?:postgresql|postgres|mysql|sqlite|mssql|oracle)'
            r'(?:\+\w+)?'
            r'://\S+'
        )
        for msg in state.get("messages", []):
            if msg.get("role") != "user":
                continue
            match = conn_pattern.search(msg.get("content", ""))
            if match:
                # Strip trailing punctuation that isn't part of the URL
                conn = match.group(0).rstrip('.,;!?)\'"')
                return conn
        return None
    
    @staticmethod
    def _keyword_in_text(keyword: str, text: str) -> bool:
        """Check if *keyword* appears in *text* as a whole word/phrase.

        Short metric-like tokens (e.g. "roc", "auc", "mae", "r2") are
        matched with word boundaries so that "process" does not
        accidentally match "roc".  Multi-word phrases and longer tokens
        use plain substring matching, which is safe.
        """
        # Tokens that are short AND can appear as substrings in normal
        # English words need word-boundary matching.
        if len(keyword) <= 4 and keyword.isalpha():
            return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))
        return keyword in text

    def _extract_intent_from_state(self, state: PipelineState) -> Dict[str, Any]:
        """Extract intent from state messages using keyword heuristics."""
        intent = {
            "prediction_target": None,
            "entity_type": None,
            "task_type": "binary_classification",  # conservative default
            "evaluation_metric": None,  # user's stated metric (e.g., "AUROC", "MAE")
            "data_source": "database" if state.get("db_connection_string") else "csv",
            "confirmed": True,
        }

        # Classification checked FIRST because its keywords ("predict if", "whether",
        # "will make any") are more specific than regression keywords that may also
        # appear in the same sentence (e.g., "votes" in "predict if user will make any votes").
        classification_keywords = [
            "churn", "leave", "cancel", "dnf", "qualify", "will happen",
            "yes or no", "probability of", "predict if", "predict whether",
            "will make any", "will do any", "whether",
            # Metric names that imply binary classification
            "auroc", "auc", "roc_auc", "roc-auc", "roc auc",
            "f1 score", "f1-score", "accuracy", "precision and recall",
        ]
        link_prediction_keywords = [
            "recommend", "which items", "list of", "purchase list",
            "map@", "precision@", "recall@", "link prediction",
            "mean average precision",
        ]
        regression_keywords = [
            "sum", "total", "sales", "revenue", "amount", "count",
            "how much", "how many", "average", "mae", "rmse", "r2",
            "mean absolute", "root mean", "ltv", "lifetime value",
            "clicks", "popularity", "number of",
        ]

        _match = self._keyword_in_text

        for msg in state.get("messages", []):
            content = msg.get("content", "")
            content_lower = content.lower()
            if "predict" not in content_lower and "forecast" not in content_lower:
                continue

            intent["prediction_target"] = content[:200]

            # Extract user's stated evaluation metric.
            # Order matters: longer / more specific tokens first so that
            # e.g. "auroc" is tried before "auc".
            metric_map = [
                ("auroc", "AUROC"), ("roc_auc", "ROC-AUC"),
                ("roc-auc", "ROC-AUC"), ("roc auc", "ROC-AUC"),
                ("mean absolute", "MAE"), ("root mean", "RMSE"),
                ("average precision", "average_precision"),
                ("f1 score", "F1"), ("f1-score", "F1"),
                ("accuracy", "accuracy"), ("ap score", "AP"),
                ("mae", "MAE"), ("rmse", "RMSE"),
                ("r2", "R2"), ("r²", "R2"),
                ("auc", "AUC"),
                ("mean average precision", "MAP"),
                ("precision@", "precision@k"), ("recall@", "recall@k"),
                ("map@", "MAP@k"),
            ]
            for kw, metric_name in metric_map:
                if _match(kw, content_lower):
                    intent["evaluation_metric"] = metric_name
                    break

            # Check classification FIRST (more specific keywords override ambiguous ones)
            if any(_match(kw, content_lower) for kw in classification_keywords):
                intent["task_type"] = "binary_classification"
            elif any(_match(kw, content_lower) for kw in link_prediction_keywords):
                intent["task_type"] = "link_prediction"
            elif any(_match(kw, content_lower) for kw in regression_keywords):
                intent["task_type"] = "regression"
            # else: keep default "binary_classification"
            break

        return intent
