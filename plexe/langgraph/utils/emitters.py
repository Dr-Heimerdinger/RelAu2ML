import asyncio
import contextvars
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from plexe.langgraph.utils.logging_utils import log_session_event

logger = logging.getLogger(__name__)

# Context variable to make the current emitter accessible from within tools
current_emitter_var: contextvars.ContextVar[Optional['BaseEmitter']] = contextvars.ContextVar(
    'current_emitter', default=None
)


def get_current_emitter() -> Optional['BaseEmitter']:
    """Get the current emitter from context (usable from within tool functions)."""
    return current_emitter_var.get()


def set_current_emitter(emitter: Optional['BaseEmitter']):
    """Set the current emitter in context."""
    current_emitter_var.set(emitter)


class BaseEmitter(ABC):
    """Base class for emitting agent thoughts and progress."""
    
    @abstractmethod
    def emit_thought(self, agent_name: str, thought: str, token_usage: Optional[Dict[str, int]] = None):
        """Emit a thinking/progress message with optional token usage."""
        pass
    
    @abstractmethod
    def emit_agent_start(self, agent_name: str, model_id: str = ""):
        """Emit agent start notification."""
        pass
    
    @abstractmethod
    def emit_agent_end(self, agent_name: str, result: str):
        """Emit agent completion notification."""
        pass
    
    @abstractmethod
    def emit_tool_call(self, agent_name: str, tool_name: str, args: Dict[str, Any]):
        """Emit tool call notification."""
        pass
    
    @abstractmethod
    def emit_tool_result(self, agent_name: str, tool_name: str, result: str):
        """Emit tool result notification."""
        pass

    def emit_training_progress(self, agent_name: str, progress_data: Dict[str, Any]):
        """Emit training progress update (epoch, loss, metrics, etc.).

        Args:
            agent_name: Name of the agent
            progress_data: Dict with keys like:
                - phase: "preparing" | "embedding" | "training" | "evaluating" | "completed"
                - current_epoch: Current epoch number
                - total_epochs: Total epochs
                - loss: Current training loss
                - metrics: Validation metrics dict
                - best_metric_name: Name of the metric being optimized
                - best_metric_value: Best value so far
                - is_best: Whether this epoch achieved best metric
                - message: Human-readable status message
                - epoch_history: List of {epoch, loss, metrics} for chart
        """
        pass  # Default no-op; subclasses override

    def emit_token_update(self, agent_name: str, cumulative: Dict[str, Any]):
        """Emit cumulative token usage update for real-time UI display.

        Args:
            agent_name: Name of the agent that triggered the update
            cumulative: Dict with keys: total, total_input_tokens,
                total_output_tokens, budget, per_agent
        """
        pass  # Default no-op; subclasses override


class ConsoleEmitter(BaseEmitter):
    """Console-based emitter for development/debugging with rich formatting."""
    
    def __init__(self):
        self.step_count = 0
    
    def emit_thought(self, agent_name: str, thought: str, token_usage: Optional[Dict[str, int]] = None):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        token_info = ""
        if token_usage:
            token_info = f" [tokens: {token_usage.get('total_tokens', 0)}]"
        print(f"[{agent_name}] Step {self.step_count} @ {timestamp}{token_info}")
        print(f"  {thought}")
    
    def emit_agent_start(self, agent_name: str, model_id: str = ""):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        model_info = f" (using {model_id})" if model_id else ""
        print(f"\n=== {agent_name} Starting{model_info} === (Step {self.step_count} @ {timestamp})")
    
    def emit_agent_end(self, agent_name: str, result: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"=== {agent_name} Completed === (Step {self.step_count} @ {timestamp})")
        if result:
            print(f"  Result: {result}")
    
    def emit_tool_call(self, agent_name: str, tool_name: str, args: Dict[str, Any]):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        args_str = ""
        if args:
            try:
                import json
                args_str = f" with args: {json.dumps(args)[:100]}"
            except:
                pass
        print(f"[{agent_name}] Step {self.step_count} @ {timestamp}")
        print(f"  Calling tool: {tool_name}{args_str}")
    
    def emit_tool_result(self, agent_name: str, tool_name: str, result: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{agent_name}] Step {self.step_count} @ {timestamp}")
        # Format with newlines for better readability
        formatted_result = result.replace('\\n', '\n') if result else ""
        print(f"  Tool result:\n{formatted_result}")

    def emit_training_progress(self, agent_name: str, progress_data: Dict[str, Any]):
        phase = progress_data.get("phase", "training")
        epoch = progress_data.get("current_epoch", 0)
        total = progress_data.get("total_epochs", 0)
        loss = progress_data.get("loss")
        msg = progress_data.get("message", "")
        loss_str = f" Loss={loss:.4f}" if loss is not None else ""
        if epoch and total:
            print(f"[{agent_name}] Training [{phase}] Epoch {epoch}/{total}{loss_str} - {msg}")
        else:
            print(f"[{agent_name}] Training [{phase}] {msg}")

    def emit_token_update(self, agent_name: str, cumulative: Dict[str, Any]):
        total = cumulative.get("total", 0)
        budget = cumulative.get("budget")
        budget_str = f" / {budget:,}" if budget else ""
        print(f"[TokenTracker] Cumulative: {total:,}{budget_str} tokens (after {agent_name})")


class WebSocketEmitter(BaseEmitter):
    """WebSocket-based emitter for UI integration with session logging."""
    
    def __init__(self, websocket, loop: Optional[asyncio.AbstractEventLoop] = None, model_id: str = ""):
        self.websocket = websocket
        self.loop = loop
        self.is_closed = False
        self.step_count = 0
        self.model_id = model_id
    
    def set_model_id(self, model_id: str):
        """Set the current model ID for context."""
        self.model_id = model_id
    
    def close(self):
        """Mark the emitter as closed."""
        self.is_closed = True
    
    def _send_message(self, message: Dict[str, Any]):
        """Send a message to the WebSocket."""
        if self.is_closed:
            return
        
        try:
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send_json(message),
                    self.loop
                )
            else:
                asyncio.get_event_loop().run_until_complete(
                    self.websocket.send_json(message)
                )
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
    
    def emit_thought(self, agent_name: str, thought: str, token_usage: Optional[Dict[str, int]] = None):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        message_data = {
            "type": "thinking",
            "role": "thinking",
            "event_type": "thinking",
            "agent_name": agent_name,
            "message": thought,
            "step_number": self.step_count,
            "timestamp": timestamp,
        }
        if token_usage:
            message_data["token_usage"] = token_usage
        self._send_message(message_data)
        # Log to session file
        token_log = f" [tokens: {token_usage}]" if token_usage else ""
        log_session_event("thinking", f"{thought}{token_log}", agent_name)
    
    def emit_agent_start(self, agent_name: str, model_id: str = ""):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        model_info = model_id or self.model_id
        message = f"Starting {agent_name}" + (f" (using {model_info})" if model_info else "")
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "agent_start",
            "agent_name": agent_name,
            "model_id": model_info,
            "message": message,
            "step_number": self.step_count,
            "timestamp": timestamp,
        })
        # Log to session file
        log_session_event("agent_start", message, agent_name, {"model": model_info})
    
    def emit_agent_end(self, agent_name: str, result: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"Completed: {result}" if result else "Completed"
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "agent_end",
            "agent_name": agent_name,
            "message": message,
            "step_number": self.step_count,
            "timestamp": timestamp,
        })
        # Log to session file (truncate for file log only)
        log_message = f"Completed: {result[:300]}" if result else "Completed"
        log_session_event("agent_end", log_message, agent_name)
    
    def emit_tool_call(self, agent_name: str, tool_name: str, args: Dict[str, Any]):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        args_str = ""
        if args:
            try:
                import json
                args_str = f" with {json.dumps(args)[:100]}"
            except:
                pass
        message = f"Calling tool: {tool_name}{args_str}"
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "tool_call",
            "agent_name": agent_name,
            "tool_name": tool_name,
            "tool_args": args,
            "message": message,
            "step_number": self.step_count,
            "timestamp": timestamp,
        })
        # Log to session file
        log_session_event("tool_call", message, agent_name, {"tool": tool_name, "args": args})
    
    def emit_tool_result(self, agent_name: str, tool_name: str, result: str):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Format with newlines for better readability
        formatted_result = result.replace('\\n', '\n') if result else "Tool completed"
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "tool_result",
            "agent_name": agent_name,
            "tool_name": tool_name,
            "message": f"Tool result:\n{formatted_result}",
            "result": result,
            "step_number": self.step_count,
            "timestamp": timestamp,
        })
        # Log to session file (truncate result for log)
        log_session_event("tool_result", f"Result from {tool_name}: {result[:500] if result else 'empty'}", agent_name)

    def emit_training_progress(self, agent_name: str, progress_data: Dict[str, Any]):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "training_progress",
            "agent_name": agent_name,
            "progress": progress_data,
            "message": progress_data.get("message", "Training in progress..."),
            "step_number": self.step_count,
            "timestamp": timestamp,
        })
        # Log to session file
        epoch = progress_data.get("current_epoch", "")
        total = progress_data.get("total_epochs", "")
        log_session_event(
            "training_progress",
            f"Epoch {epoch}/{total}: {progress_data.get('message', '')}",
            agent_name,
        )

    def emit_token_update(self, agent_name: str, cumulative: Dict[str, Any]):
        total = cumulative.get("total", 0)
        budget = cumulative.get("budget")
        budget_str = f" / {budget:,}" if budget else ""
        self._send_message({
            "type": "thinking",
            "role": "thinking",
            "event_type": "token_update",
            "agent_name": agent_name,
            "cumulative_tokens": cumulative,
            "message": f"Tokens used: {total:,}{budget_str}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        log_session_event(
            "token_update",
            f"Cumulative: {total:,}{budget_str} tokens (after {agent_name})",
            agent_name,
        )


class MultiEmitter(BaseEmitter):
    """Combines multiple emitters."""
    
    def __init__(self, emitters: List[BaseEmitter]):
        self.emitters = emitters
    
    def emit_thought(self, agent_name: str, thought: str):
        for emitter in self.emitters:
            try:
                emitter.emit_thought(agent_name, thought)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
    
    def emit_agent_start(self, agent_name: str, model_id: str = ""):
        for emitter in self.emitters:
            try:
                emitter.emit_agent_start(agent_name, model_id)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
    
    def emit_agent_end(self, agent_name: str, result: str):
        for emitter in self.emitters:
            try:
                emitter.emit_agent_end(agent_name, result)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
    
    def emit_tool_call(self, agent_name: str, tool_name: str, args: Dict[str, Any]):
        for emitter in self.emitters:
            try:
                emitter.emit_tool_call(agent_name, tool_name, args)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
    
    def emit_tool_result(self, agent_name: str, tool_name: str, result: str):
        for emitter in self.emitters:
            try:
                emitter.emit_tool_result(agent_name, tool_name, result)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")

    def emit_training_progress(self, agent_name: str, progress_data: Dict[str, Any]):
        for emitter in self.emitters:
            try:
                emitter.emit_training_progress(agent_name, progress_data)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")

    def emit_token_update(self, agent_name: str, cumulative: Dict[str, Any]):
        for emitter in self.emitters:
            try:
                emitter.emit_token_update(agent_name, cumulative)
            except Exception as e:
                logger.warning(f"Emitter error: {e}")
