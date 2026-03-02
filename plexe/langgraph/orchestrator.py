"""
Plexe Orchestrator using LangGraph.

This module provides the main orchestrator that coordinates
all agents in the ML pipeline using LangGraph's StateGraph.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Literal

from plexe.langgraph.utils.logging_utils import session_id_var

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from plexe.langgraph.state import (
    PipelineState,
    PipelinePhase,
    create_initial_state,
)
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.utils import BaseEmitter, ConsoleEmitter, ChainOfThoughtCallback
from plexe.langgraph.agents import (
    ConversationalAgent,
    EDAAgent,
    DatasetBuilderAgent,
    TaskBuilderAgent,
    RelationalGNNSpecialistAgent,
    OperationAgent,
)

logger = logging.getLogger(__name__)


class PlexeOrchestrator:
    """
    Main orchestrator for the Plexe ML pipeline using LangGraph.
    
    This orchestrator manages the workflow between specialized agents:
    1. ConversationalAgent - User interaction and requirements gathering
    2. EDAAgent - Schema analysis, data export, and exploratory data analysis
    3. DatasetBuilderAgent - Dataset class generation
    4. TaskBuilderAgent - Task class and SQL generation
    5. RelationalGNNSpecialistAgent - GNN training
    6. OperationAgent - Environment and execution management
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        verbose: bool = False,
        callback: Optional[Callable] = None,
        emitter: Optional[BaseEmitter] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Agent configuration (uses defaults if None)
            verbose: Enable verbose logging
            callback: Optional callback for progress updates
            emitter: Optional emitter for UI communication
        """
        self.config = config or AgentConfig.from_env()
        self.verbose = verbose
        self.callback = callback
        self.emitter = emitter or ConsoleEmitter()
        
        self._init_agents()
        self._build_graph()
    
    def set_emitter(self, emitter: BaseEmitter):
        """Set the emitter for all agents."""
        self.emitter = emitter
        for agent in [
            self.conversational_agent,
            self.eda_agent,
            self.dataset_builder_agent,
            self.task_builder_agent,
            self.gnn_specialist_agent,
            self.operation_agent,
        ]:
            agent.set_emitter(emitter)
    
    def _init_agents(self):
        """Initialize all agents."""
        self.conversational_agent = ConversationalAgent(config=self.config)
        self.eda_agent = EDAAgent(config=self.config)
        self.dataset_builder_agent = DatasetBuilderAgent(config=self.config)
        self.task_builder_agent = TaskBuilderAgent(config=self.config)
        self.gnn_specialist_agent = RelationalGNNSpecialistAgent(config=self.config)
        self.operation_agent = OperationAgent(config=self.config)
        
        for agent in [
            self.conversational_agent,
            self.eda_agent,
            self.dataset_builder_agent,
            self.task_builder_agent,
            self.gnn_specialist_agent,
            self.operation_agent,
        ]:
            agent.set_emitter(self.emitter)
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(PipelineState)
        
        workflow.add_node("conversation", self._conversation_node)
        workflow.add_node("schema_analysis", self._schema_analysis_node)
        workflow.add_node("dataset_building", self._dataset_building_node)
        workflow.add_node("task_building", self._task_building_node)
        workflow.add_node("gnn_training", self._gnn_training_node)
        workflow.add_node("operation", self._operation_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        workflow.set_entry_point("conversation")
        
        workflow.add_conditional_edges(
            "conversation",
            self._route_from_conversation,
            {
                "continue": "conversation",
                "proceed": "schema_analysis",
                "end": END,
            }
        )
        
        workflow.add_conditional_edges(
            "schema_analysis",
            self._route_from_schema,
            {
                "success": "dataset_building",
                "error": "error_handler",
            }
        )
        
        workflow.add_conditional_edges(
            "dataset_building",
            self._route_from_dataset,
            {
                "success": "task_building",
                "error": "error_handler",
            }
        )
        
        workflow.add_conditional_edges(
            "task_building",
            self._route_from_task,
            {
                "success": "gnn_training",
                "error": "error_handler",
            }
        )
        
        workflow.add_conditional_edges(
            "gnn_training",
            self._route_from_training,
            {
                "success": "operation",
                "error": "error_handler",
            }
        )
        
        workflow.add_edge("operation", END)
        
        workflow.add_conditional_edges(
            "error_handler",
            self._route_from_error,
            {
                "retry": "conversation",
                "end": END,
            }
        )
        
        self.checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=self.checkpointer)
    
    def _conversation_node(self, state: PipelineState) -> Dict[str, Any]:
        """Handle conversation with user."""
        self._log_phase("Conversation", "ConversationalAgent")
        result = self.conversational_agent.invoke(state)
        return result
    
    def _schema_analysis_node(self, state: PipelineState) -> Dict[str, Any]:
        """Handle schema analysis, data export, and EDA."""
        self._log_phase("Schema Analysis & EDA", "EDAAgent")
        result = self.eda_agent.invoke(state)
        return result
    
    def _dataset_building_node(self, state: PipelineState) -> Dict[str, Any]:
        """Handle dataset class generation."""
        self._log_phase("Dataset Building", "DatasetBuilderAgent")
        result = self.dataset_builder_agent.invoke(state)
        return result
    
    def _task_building_node(self, state: PipelineState) -> Dict[str, Any]:
        """Handle task class generation."""
        self._log_phase("Task Building", "TaskBuilderAgent")
        result = self.task_builder_agent.invoke(state)
        return result
    
    def _gnn_training_node(self, state: PipelineState) -> Dict[str, Any]:
        """Handle GNN training."""
        self._log_phase("GNN Training", "RelationalGNNSpecialistAgent")
        result = self.gnn_specialist_agent.invoke(state)
        return result
    
    def _operation_node(self, state: PipelineState) -> Dict[str, Any]:
        """Handle operation and finalization."""
        self._log_phase("Operation", "OperationAgent")
        result = self.operation_agent.invoke(state)
        result["current_phase"] = PipelinePhase.COMPLETED.value
        return result
    
    def _error_handler_node(self, state: PipelineState) -> Dict[str, Any]:
        """Handle errors and determine recovery strategy."""
        errors = state.get("errors", [])
        logger.error(f"Pipeline error: {errors}")
        
        if self.emitter:
            self.emitter.emit_thought("ErrorHandler", f"Handling errors: {errors}")
        
        # Safely get retry_count - handle None case explicitly
        metadata = state.get("metadata") or {}
        retry_count = metadata.get("retry_count", 0)
        max_retries = self.config.max_retries
        
        if retry_count < max_retries:
            return {
                "metadata": {**metadata, "retry_count": retry_count + 1},
                "warnings": [f"Retrying after error (attempt {retry_count + 1}/{max_retries})"],
            }
        
        return {
            "current_phase": PipelinePhase.FAILED.value,
        }
    
    def _route_from_conversation(self, state: PipelineState) -> Literal["continue", "proceed", "end"]:
        """Route from conversation node."""
        logger.debug(f"Routing from conversation: user_confirmation_required={state.get('user_confirmation_required')}, "
                    f"user_confirmed={state.get('user_confirmed')}, user_intent={state.get('user_intent')}, "
                    f"db_connection_string={bool(state.get('db_connection_string'))}")
        
        if state.get("user_confirmation_required"):
            if state.get("user_confirmed"):
                logger.info("User confirmed, proceeding to schema analysis")
                return "proceed"
            return "continue"
        
        if state.get("db_connection_string") or state.get("csv_dir"):
            if state.get("user_intent"):
                logger.info("Intent detected with data source, proceeding to schema analysis")
                return "proceed"
        
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                ready_indicators = [
                    "ready to proceed",
                    "start building",
                    "begin training",
                    "let's begin",
                    "i'll start",
                    "proceed with",
                    "starting the pipeline",
                ]
                if any(indicator in content for indicator in ready_indicators):
                    if state.get("db_connection_string") or state.get("csv_dir"):
                        logger.info("Ready indicator found in response, proceeding")
                        return "proceed"
                break
        
        return "continue"
    
    def _route_from_schema(self, state: PipelineState) -> Literal["success", "error"]:
        """Route from schema analysis node."""
        if state.get("errors"):
            return "error"
        if state.get("csv_dir") and state.get("schema_info"):
            return "success"
        if state.get("csv_dir"):
            return "success"
        return "error"
    
    def _route_from_dataset(self, state: PipelineState) -> Literal["success", "error"]:
        """Route from dataset building node."""
        if state.get("errors"):
            return "error"
        
        # Check if dataset_info exists AND has no error
        dataset_info = state.get("dataset_info")
        if dataset_info:
            # If dataset_info has error field, treat as failure
            if dataset_info.get("error"):
                logger.error(f"Dataset building failed: {dataset_info.get('error')}")
                return "error"
        
        # Also verify the actual file exists
        working_dir = state.get("working_dir", "")
        dataset_path = os.path.join(working_dir, "dataset.py")
        if os.path.exists(dataset_path):
            return "success"
        
        logger.error(f"Dataset file not found at: {dataset_path}")
        return "error"
    
    def _route_from_task(self, state: PipelineState) -> Literal["success", "error"]:
        """Route from task building node."""
        if state.get("errors"):
            return "error"
        
        # Check if task_info exists AND has no error
        task_info = state.get("task_info")
        if task_info:
            # If task_info has error field, treat as failure
            if task_info.get("error"):
                logger.error(f"Task building failed: {task_info.get('error')}")
                return "error"
        
        # Also verify the actual file exists
        working_dir = state.get("working_dir", "")
        task_path = os.path.join(working_dir, "task.py")
        if os.path.exists(task_path):
            return "success"
        
        logger.error(f"Task file not found at: {task_path}")
        return "error"
    
    def _route_from_training(self, state: PipelineState) -> Literal["success", "error"]:
        """Route from GNN training node to operation.

        The GNN Specialist generates a training script and sets
        ``training_script_ready``.  Actual training execution is handled
        by the Operation Agent downstream.
        """
        if state.get("errors"):
            return "error"
        if state.get("training_script_ready"):
            return "success"
        working_dir = state.get("working_dir", "")
        if os.path.exists(os.path.join(working_dir, "train_script.py")):
            return "success"
        if state.get("training_result"):
            return "success"
        return "error"
    
    def _route_from_error(self, state: PipelineState) -> Literal["retry", "end"]:
        """Route from error handler."""
        # Safely get retry_count - handle None case explicitly
        metadata = state.get("metadata") or {}
        retry_count = metadata.get("retry_count", 0)
        if retry_count < self.config.max_retries:
            return "retry"
        return "end"
    
    def _log_phase(self, phase: str, agent_name: str = ""):
        """Log phase transition with rich formatting."""
        if self.verbose:
            logger.info(f"=== Phase: {phase} ({agent_name}) ===")
        if self.callback:
            self.callback({
                "phase": phase,
                "agent": agent_name,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        if self.emitter:
            self.emitter.emit_thought("Orchestrator", f"Pipeline Phase: {phase} - Activating {agent_name}")
    
    def run(
        self,
        user_message: str,
        db_connection_string: Optional[str] = None,
        csv_dir: Optional[str] = None,
        working_dir: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the ML pipeline.
        
        Args:
            user_message: Initial user request
            db_connection_string: Optional database connection
            csv_dir: Optional directory with CSV files
            working_dir: Working directory for artifacts
            session_id: Session identifier
        
        Returns:
            Final pipeline state
        """
        if session_id is None:
            session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if working_dir is None:
            working_dir = os.path.join("workdir", session_id)
        
        os.makedirs(working_dir, exist_ok=True)
        
        # Set session ID for logging
        session_id_token = session_id_var.set(session_id)
        
        initial_state = create_initial_state(
            session_id=session_id,
            working_dir=working_dir,
            user_message=user_message,
            db_connection_string=db_connection_string,
        )
        
        if csv_dir:
            initial_state["csv_dir"] = csv_dir
        
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            final_state = self.graph.invoke(initial_state, config)
            return {
                "status": "completed" if final_state.get("current_phase") == PipelinePhase.COMPLETED.value else "failed",
                "state": final_state,
                "session_id": session_id,
                "working_dir": working_dir,
            }
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "working_dir": working_dir,
            }
    
    def chat(
        self,
        message: str,
        session_id: str,
        working_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat message to an ongoing session.
        
        Args:
            message: User message
            session_id: Existing session ID
            working_dir: Working directory
        
        Returns:
            Agent response
        """
        config = {"configurable": {"thread_id": session_id}}
        
        # Set session ID for logging
        session_id_var.set(session_id)
        
        try:
            current_state = self.graph.get_state(config)
            
            if current_state and current_state.values:
                state = dict(current_state.values)
                state["messages"] = state.get("messages", []) + [{
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.now().isoformat(),
                }]
                
                if state.get("user_confirmation_required"):
                    if any(word in message.lower() for word in ["yes", "proceed", "confirm", "ok"]):
                        state["user_confirmed"] = True
                    elif any(word in message.lower() for word in ["no", "stop", "cancel"]):
                        state["user_confirmed"] = False
                        state["current_phase"] = PipelinePhase.CONVERSATION.value
                
                result = self.graph.invoke(state, config)
                return {
                    "status": "success",
                    "response": self._extract_response(result),
                    "phase": result.get("current_phase"),
                }
            else:
                return self.run(message, working_dir=working_dir, session_id=session_id)
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _extract_response(self, state: Dict[str, Any]) -> str:
        """Extract the latest assistant response from state."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a session."""
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.graph.get_state(config)
            return dict(state.values) if state and state.values else None
        except Exception:
            return None
