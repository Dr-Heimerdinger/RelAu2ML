"""
Configuration for LangGraph-based agents.

This module provides configuration management for agent models
using environment variables and defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ExternalAPIConfig:
    """Configuration for external API services."""
    
    # Semantic Scholar API
    semantic_scholar_api_key: Optional[str] = field(default_factory=lambda: os.environ.get(
        "SEMANTIC_SCHOLAR_API_KEY"
    ))
    
    # arXiv API (no key needed, but rate limits apply)
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    
    # Papers With Code API
    papers_with_code_api_url: str = "https://paperswithcode.com/api/v1"
    
    # Hugging Face API (for datasets and models)
    huggingface_token: Optional[str] = field(default_factory=lambda: os.environ.get(
        "HF_TOKEN"
    ))
    
    # OpenML API
    openml_api_key: Optional[str] = field(default_factory=lambda: os.environ.get(
        "OPENML_API_KEY"
    ))
    
    # Rate limiting
    max_requests_per_minute: int = 10
    request_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "ExternalAPIConfig":
        """Create configuration from environment variables."""
        return cls()


@dataclass
class AgentConfig:
    """Configuration for agent models from environment variables."""
    
    orchestrator_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_ORCHESTRATOR_MODEL", "openai/gpt-4o"
    ))
    conversational_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_CONVERSATIONAL_MODEL", "openai/gpt-4o"
    ))
    eda_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_EDA_MODEL", "openai/gpt-4o"
    ))
    dataset_builder_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_DATASET_BUILDER_MODEL", "openai/gpt-4o"
    ))
    task_builder_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_TASK_BUILDER_MODEL", "openai/gpt-4o"
    ))
    gnn_specialist_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_GNN_SPECIALIST_MODEL", "openai/gpt-4o"
    ))
    # NOTE: operation_model is kept for backward compatibility but is no longer
    # used. The OperationAgent now runs as a pure deterministic function (no LLM).
    operation_model: str = field(default_factory=lambda: os.environ.get(
        "PLEXE_OPERATION_MODEL", "openai/gpt-4o"
    ))
    
    temperature: float = field(default_factory=lambda: float(os.environ.get(
        "PLEXE_AGENT_TEMPERATURE", "0.1"
    )))
    max_context_messages: int = field(default_factory=lambda: int(os.environ.get(
        "PLEXE_MAX_CONTEXT_MESSAGES", "10"
    )))
    token_budget: Optional[int] = field(default_factory=lambda: (
        int(os.environ["PLEXE_TOKEN_BUDGET"])
        if os.environ.get("PLEXE_TOKEN_BUDGET")
        else None
    ))
    max_retries: int = field(default_factory=lambda: int(os.environ.get(
        "PLEXE_MAX_RETRIES", "3"
    )))
    retry_config: Dict[str, int] = field(default_factory=lambda: {
        "conversation": 1,
        "schema_analysis": 3,
        "dataset_building": 2,
        "task_building": 2,
        "gnn_training": 1,
        "operation": 2,
    })
    verbose: bool = field(default_factory=lambda: os.environ.get(
        "PLEXE_VERBOSE", "false"
    ).lower() == "true")
    early_stopping: bool = field(default_factory=lambda: os.environ.get(
        "PLEXE_EARLY_STOPPING", "false"
    ).lower() == "true")
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables."""
        return cls()
    
    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the model ID for a specific agent type."""
        mapping = {
            "orchestrator": self.orchestrator_model,
            "conversational": self.conversational_model,
            "eda": self.eda_model,
            "dataset_builder": self.dataset_builder_model,
            "task_builder": self.task_builder_model,
            "gnn_specialist": self.gnn_specialist_model,
            "operation": self.operation_model,
        }
        return mapping.get(agent_type, self.orchestrator_model)


def get_llm_from_model_id(model_id: str, temperature: float = 0.1):
    """
    Create a LangChain LLM instance from a model ID string.
    
    Supports formats:
    - openai/gpt-4o -> OpenAI
    - anthropic/claude-sonnet-4-20250514 -> Anthropic
    - gemini/gemini-2.5-flash -> Google Gemini
    """
    from langchain_core.language_models import BaseChatModel
    
    if model_id.startswith("openai/"):
        from langchain_openai import ChatOpenAI
        model_name = model_id.replace("openai/", "")
        return ChatOpenAI(model=model_name, temperature=temperature)
    
    elif model_id.startswith("anthropic/"):
        from langchain_anthropic import ChatAnthropic
        model_name = model_id.replace("anthropic/", "")
        return ChatAnthropic(model=model_name, temperature=temperature)
    
    elif model_id.startswith("gemini/"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = model_id.replace("gemini/", "")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=temperature)
