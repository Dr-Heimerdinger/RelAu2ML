"""
Pipeline state management for LangGraph workflow.

This module defines the shared state that flows through the
multi-agent pipeline using TypedDict for LangGraph compatibility.
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from enum import Enum
import operator


class PipelinePhase(str, Enum):
    """Phases of the ML pipeline."""
    CONVERSATION = "conversation"
    SCHEMA_ANALYSIS = "schema_analysis"
    DATASET_BUILDING = "dataset_building"
    TASK_BUILDING = "task_building"
    GNN_TRAINING = "gnn_training"
    OPERATION = "operation"
    COMPLETED = "completed"
    FAILED = "failed"


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    RECOVERABLE = "recoverable"


class ErrorRecord(TypedDict, total=False):
    agent: str
    phase: str
    category: str
    message: str
    exception_type: str
    timestamp: str
    retry_attempt: int


class MessageRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(TypedDict):
    """A single message in the conversation."""
    role: str
    content: str
    timestamp: Optional[str]


class DatasetInfo(TypedDict, total=False):
    """Information about the dataset."""
    name: str
    file_path: str
    class_name: str
    val_timestamp: str
    test_timestamp: str
    tables: List[str]
    csv_dir: str


class TaskInfo(TypedDict, total=False):
    """Information about the prediction task."""
    name: str
    file_path: str
    class_name: str
    task_type: str
    entity_table: str
    target_column: str
    metrics: List[str]


class SchemaInfo(TypedDict, total=False):
    """Database schema information."""
    tables: Dict[str, Any]
    relationships: List[Dict[str, str]]
    temporal_columns: Dict[str, str]
    primary_keys: Dict[str, str]
    foreign_keys: Dict[str, List[Dict[str, str]]]


class EDAInfo(TypedDict, total=False):
    """Exploratory Data Analysis information."""
    statistics: Dict[str, Any]
    quality_issues: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    relationship_analysis: Dict[str, Any]
    summary: Dict[str, Any]


class TrainingResult(TypedDict, total=False):
    """Results from GNN training."""
    metrics: Dict[str, float]
    best_epoch: int
    model_path: str
    training_time: float
    script_path: str


class PipelineState(TypedDict, total=False):
    """
    Shared state for the LangGraph pipeline.
    
    This state is passed between agents and accumulates information
    as the pipeline progresses through different phases.
    """
    session_id: str
    working_dir: str
    current_phase: str
    
    messages: Annotated[List[Message], operator.add]
    user_intent: str
    
    db_connection_string: Optional[str]
    csv_dir: Optional[str]
    
    schema_info: Optional[SchemaInfo]
    eda_info: Optional[EDAInfo]
    csv_files_info: Optional[Dict[str, Any]]
    dataset_info: Optional[DatasetInfo]
    task_info: Optional[TaskInfo]
    training_result: Optional[TrainingResult]
    training_script_ready: Optional[bool]
    training_script_path: Optional[str]
    selected_hyperparameters: Optional[Dict[str, Any]]
    
    generated_code: Dict[str, str]
    artifacts: List[str]

    active_errors: List[str]
    error_records: List[ErrorRecord]
    error_history: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]
    failed_phase: Optional[str]

    user_confirmation_required: bool
    user_confirmation_context: Optional[Dict[str, Any]]
    user_confirmed: Optional[bool]

    metadata: Dict[str, Any]
    token_usage_summary: Optional[Dict[str, Any]]


def create_initial_state(
    session_id: str,
    working_dir: str,
    user_message: str,
    db_connection_string: Optional[str] = None,
) -> PipelineState:
    """
    Create the initial pipeline state.
    
    Args:
        session_id: Unique identifier for this session
        working_dir: Working directory for artifacts
        user_message: Initial user message/request
        db_connection_string: Optional database connection string
    
    Returns:
        Initial pipeline state
    """
    from datetime import datetime
    
    return PipelineState(
        session_id=session_id,
        working_dir=working_dir,
        current_phase=PipelinePhase.CONVERSATION.value,
        messages=[{
            "role": MessageRole.USER.value,
            "content": user_message,
            "timestamp": datetime.now().isoformat(),
        }],
        user_intent="",
        db_connection_string=db_connection_string,
        csv_dir=None,
        schema_info=None,
        eda_info=None,
        csv_files_info=None,
        dataset_info=None,
        task_info=None,
        training_result=None,
        selected_hyperparameters=None,
        generated_code={},
        artifacts=[],
        active_errors=[],
        error_records=[],
        error_history=[],
        warnings=[],
        failed_phase=None,
        user_confirmation_required=False,
        user_confirmation_context=None,
        user_confirmed=None,
        metadata={},
        token_usage_summary=None,
    )
