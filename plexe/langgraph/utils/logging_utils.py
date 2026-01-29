import logging
import os
from pathlib import Path
from contextvars import ContextVar
from typing import Optional, Dict
from datetime import datetime

# Context variable to store the session ID for the current thread/context
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)

# Get the project root directory (where logs should be created)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


class SessionLogger:
    """
    Logger for session-specific experiment logs.
    
    Only logs activities within a chat session/experiment,
    not general backend infrastructure logs.
    """
    
    _handlers: Dict[str, logging.FileHandler] = {}
    
    @classmethod
    def get_session_logger(cls, session_id: str) -> logging.Logger:
        """Get or create a logger for a specific session."""
        logger_name = f"plexe.session.{session_id}"
        session_logger = logging.getLogger(logger_name)
        
        if session_id not in cls._handlers:
            log_file = LOG_DIR / f"session-{session_id}.log"
            handler = logging.FileHandler(str(log_file), encoding='utf-8')
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            
            session_logger.addHandler(handler)
            session_logger.setLevel(logging.INFO)
            session_logger.propagate = False  # Don't propagate to root logger
            
            cls._handlers[session_id] = handler
            
        return session_logger
    
    @classmethod
    def close_session(cls, session_id: str):
        """Close and remove the handler for a session."""
        if session_id in cls._handlers:
            cls._handlers[session_id].close()
            del cls._handlers[session_id]
            
            # Also remove the logger
            logger_name = f"plexe.session.{session_id}"
            if logger_name in logging.Logger.manager.loggerDict:
                del logging.Logger.manager.loggerDict[logger_name]


def log_session_event(
    event_type: str,
    message: str,
    agent_name: str = "",
    extra: Optional[Dict] = None
):
    """
    Log an event to the current session's log file.
    
    Args:
        event_type: Type of event (e.g., 'agent_start', 'tool_call', 'thinking')
        message: The log message
        agent_name: Name of the agent (optional)
        extra: Additional data to log (optional)
    """
    session_id = session_id_var.get()
    if not session_id:
        return  # No session, don't log
    
    session_logger = SessionLogger.get_session_logger(session_id)
    
    # Format the log message
    parts = []
    if agent_name:
        parts.append(f"[{agent_name}]")
    parts.append(f"[{event_type}]")
    parts.append(message)
    
    log_message = " ".join(parts)
    
    if extra:
        log_message += f" | {extra}"
    
    session_logger.info(log_message)


def setup_session_logging(level=logging.INFO):
    """
    Initialize session logging infrastructure.
    
    This no longer modifies the root logger - it just ensures
    the log directory exists and the infrastructure is ready.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Session logging initialized. Logs will be written to: {LOG_DIR}")
