"""
Base agent class for LangGraph agents.

Provides common functionality for all specialized agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents import create_agent

from plexe.langgraph.config import AgentConfig, get_llm_from_model_id
from plexe.langgraph.state import PipelineState, ErrorCategory
from plexe.langgraph.utils import BaseEmitter, ChainOfThoughtCallback
from plexe.langgraph.mcp_manager import MCPManager

logger = logging.getLogger(__name__)

def extract_text_content(content) -> str:
    """Extract text from message content (handles string or list format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for agent events with detailed chain-of-thought."""
    
    def __init__(self, agent_name: str, emitter: Optional[BaseEmitter] = None, model_id: str = ""):
        self.agent_name = agent_name
        self.emitter = emitter
        self.model_id = model_id
        self.current_thought = ""
        self._llm_start_emitted = False
        self._last_emitted_text = ""  # Track to avoid duplicate emissions
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Don't emit "Thinking..." - wait for actual response
        self._llm_start_emitted = True
    
    def on_llm_end(self, response, **kwargs):
        if not self.emitter or not response or not response.generations:
            return
        
        try:
            generation = response.generations[0][0]
            text = None
            thinking_text = None
            
            # Extract thinking/reasoning from extended thinking models (Claude, etc.)
            if hasattr(generation, 'message'):
                message = generation.message
                # Check for thinking blocks (extended thinking)
                if hasattr(message, 'content') and isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, dict):
                            if block.get("type") == "thinking":
                                thinking_text = block.get("thinking", "")
                            elif block.get("type") == "text":
                                text = block.get("text", "")
                elif hasattr(message, 'content'):
                    text = extract_text_content(message.content)
                
                # Also check additional_kwargs for reasoning (Gemini, DeepSeek, etc.)
                if hasattr(message, 'additional_kwargs'):
                    kwargs_data = message.additional_kwargs
                    # DeepSeek reasoning
                    if 'reasoning_content' in kwargs_data:
                        thinking_text = kwargs_data['reasoning_content']
                    # Gemini thinking in response_metadata
                    if hasattr(message, 'response_metadata'):
                        metadata = message.response_metadata or {}
                        # Some models put thinking in candidates
                        if 'candidates' in metadata:
                            for candidate in metadata.get('candidates', []):
                                if 'content' in candidate:
                                    parts = candidate['content'].get('parts', [])
                                    for part in parts:
                                        if part.get('thought'):
                                            thinking_text = part.get('text', '')
            
            # Fallback to text extraction
            if not text and hasattr(generation, 'text') and generation.text:
                text = extract_text_content(generation.text)
            
            # Determine what to emit - prefer thinking, fallback to text
            emit_text = thinking_text or text
            if not emit_text:
                return
            
            # Avoid duplicate emissions
            emit_text = emit_text.strip()
            if emit_text == self._last_emitted_text:
                return
            self._last_emitted_text = emit_text
            
            # Skip if it looks like a tool call response (starts with JSON or action)
            if emit_text.startswith('{') or emit_text.startswith('Action:'):
                return
            
            # Use full text without truncation for UI display
            display_text = emit_text
            
            self.current_thought = display_text
            model_info = f" [{self.model_id}]" if self.model_id else ""
            
            # Extract token usage if available
            token_usage = None
            if hasattr(response, 'llm_output') and response.llm_output:
                # Some LLMs return token_usage in llm_output
                token_usage = response.llm_output.get('token_usage')
            elif hasattr(generation, 'message'):
                message = generation.message
                # Check for usage_metadata (LangChain standard as of 0.2+)
                if hasattr(message, 'usage_metadata') and message.usage_metadata:
                    usage = message.usage_metadata
                    # usage_metadata is a TypedDict with keys: input_tokens, output_tokens, total_tokens
                    token_usage = {
                        'prompt_tokens': usage.get('input_tokens', 0),
                        'completion_tokens': usage.get('output_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0)
                    }
                # Fallback: Check response_metadata for usage info (older format)
                elif hasattr(message, 'response_metadata') and message.response_metadata:
                    metadata = message.response_metadata
                    if 'usage' in metadata:
                        usage = metadata['usage']
                        token_usage = {
                            'prompt_tokens': usage.get('prompt_tokens', 0),
                            'completion_tokens': usage.get('completion_tokens', 0),
                            'total_tokens': usage.get('total_tokens', 0)
                        }
            
            if thinking_text:
                self.emitter.emit_thought(self.agent_name, f"💭 Reasoning{model_info}:\n{display_text}", token_usage)
            else:
                self.emitter.emit_thought(self.agent_name, f"💡 Analysis{model_info}:\n{display_text}", token_usage)
                
        except Exception as e:
            logger.debug(f"Error extracting LLM response: {e}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        if self.emitter:
            tool_name = serialized.get("name", "tool") if isinstance(serialized, dict) else "tool"
            args = {}
            if isinstance(input_str, str):
                try:
                    import json
                    args = json.loads(input_str) if input_str.startswith("{") else {"input": input_str[:100]}
                except:
                    args = {"input": str(input_str)[:100]}
            elif isinstance(input_str, dict):
                args = {k: str(v)[:50] for k, v in list(input_str.items())[:3]}
            self.emitter.emit_tool_call(self.agent_name, tool_name, args)
    
    def on_tool_end(self, output, **kwargs):
        if self.emitter and output:
            output_str = str(output) if output else ""
            if output_str:
                # Format with newlines for better readability
                formatted_output = output_str.replace('\\n', '\n')
                self.emitter.emit_thought(self.agent_name, f"Tool result:\n{formatted_output}")
    
    def on_chain_error(self, error, **kwargs):
        if self.emitter:
            self.emitter.emit_thought(self.agent_name, f"Error encountered: {str(error)[:200]}")


class BaseAgent(ABC):
    """Base class for all LangGraph agents."""
    
    def __init__(
        self,
        agent_type: str,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[BaseTool]] = None,
        emitter: Optional[BaseEmitter] = None,
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_type: Type identifier for this agent
            config: Agent configuration (uses defaults if None)
            tools: List of tools available to this agent
            emitter: Optional emitter for progress callbacks
        """
        self.agent_type = agent_type
        self.config = config or AgentConfig.from_env()
        self.emitter = emitter
        self.tools = tools or []
        
        self.mcp_manager = MCPManager()
        try:
            self.mcp_manager.initialize_sync(timeout=30)
            mcp_tools = self.mcp_manager.get_tools()
            if mcp_tools:
                logger.info(f"Agent {self.name} loaded {len(mcp_tools)} MCP tools")
                self.tools.extend(mcp_tools)
        except Exception as e:
            logger.warning(f"Could not load MCP tools for {self.name}: {e}")
            
        self.model_id = self.config.get_model_for_agent(agent_type)
        self.llm = get_llm_from_model_id(self.model_id, self.config.temperature)
        
        self._agent = None
        self._callback_handler = AgentCallbackHandler(self.name, emitter, self.model_id)
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
    
    @property
    def name(self) -> str:
        """Return the agent name."""
        return self.__class__.__name__
    
    @property
    def description(self) -> str:
        """Return the agent description."""
        return self.system_prompt[:200] + "..."
    
    def set_emitter(self, emitter: BaseEmitter):
        """Set the emitter for progress callbacks."""
        self.emitter = emitter
        self._callback_handler = AgentCallbackHandler(self.name, emitter, self.model_id)
    
    def get_agent(self):
        """Get or create the LangGraph agent."""
        if self._agent is None:
            self._agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=self.system_prompt,
            )
        return self._agent
    
    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        """
        Invoke the agent with the current state, streaming thoughts to emitter.
        
        Args:
            state: Current pipeline state
        
        Returns:
            Updated state components
        """
        agent = self.get_agent()
        
        messages = self._build_messages(state)
        logger.info(f"Agent {self.name} invoking with {len(messages)} messages using model {self.model_id}")
        
        if self.emitter:
            self.emitter.emit_agent_start(self.name, self.model_id)
        
        try:
            config = {"callbacks": [self._callback_handler]} if self.emitter else {}
            
            result = None
            last_valid_output = None
            for chunk in agent.stream({"messages": messages}, config=config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    # Store valid outputs (dict with messages)
                    if isinstance(node_output, dict) and node_output.get("messages"):
                        last_valid_output = node_output
                    
                    if self.emitter and node_name == "agent" and isinstance(node_output, dict):
                        agent_messages = node_output.get("messages", [])
                        for msg in agent_messages:
                            if isinstance(msg, AIMessage):
                                # Don't emit here - let the callback handler do it
                                # Just emit tool calls
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                                        tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                                        self.emitter.emit_tool_call(self.name, tool_name, tool_args)
                    result = node_output
            
            # Use last valid output if final result is None or invalid
            if result is None or not isinstance(result, dict):
                if last_valid_output:
                    result = last_valid_output
                else:
                    result = agent.invoke({"messages": messages}, config=config)
            elif not result.get("messages"):
                if last_valid_output:
                    result = last_valid_output
                else:
                    result = {"messages": []}
            
            logger.info(f"Agent {self.name} received {len(result.get('messages', []))} response messages")
            
            processed = self._process_result(result, state)
            
            if self.emitter:
                response_text = ""
                for msg in processed.get("messages", []):
                    if msg.get("role") == "assistant":
                        response_text = msg.get("content", "")
                        break
                self.emitter.emit_agent_end(self.name, response_text)
            
            return processed
        except Exception as e:
            logger.error(f"Agent {self.name} failed: {e}", exc_info=True)

            category = ErrorCategory.PERMANENT.value
            if isinstance(e, (TimeoutError, ConnectionError, ConnectionRefusedError, OSError)):
                category = ErrorCategory.TRANSIENT.value
            elif isinstance(e, (SyntaxError, ValueError)):
                category = ErrorCategory.RECOVERABLE.value
            else:
                msg_lower = str(e).lower()
                if "rate limit" in msg_lower or "429" in str(e) or "timeout" in msg_lower:
                    category = ErrorCategory.TRANSIENT.value

            from datetime import datetime
            error_msg = f"{self.name} error ({category}): {str(e)}"
            error_record = {
                "agent": self.name,
                "phase": "",
                "category": category,
                "message": str(e),
                "exception_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
            }

            if self.emitter:
                self.emitter.emit_agent_end(self.name, f"Error: {str(e)}")
            return {
                "active_errors": [error_msg],
                "error_records": [error_record],
                "error_history": [f"{self.name} [{type(e).__name__}]: {str(e)}"],
            }
    
    def _build_messages(self, state: PipelineState) -> List:
        """Build message list from state."""
        messages = [SystemMessage(content=self.system_prompt)]
        
        for msg in state.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        
        context = self._build_context(state)
        if context:
            messages.append(HumanMessage(content=f"Current context:\n{context}"))
        
        return messages
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context string from state for the agent."""
        context_parts = []
        
        if state.get("working_dir"):
            context_parts.append(f"Working directory: {state['working_dir']}")
        
        if state.get("db_connection_string"):
            context_parts.append(f"Database connection: {state['db_connection_string']}")
        
        if state.get("csv_dir"):
            context_parts.append(f"CSV directory: {state['csv_dir']}")
        
        if state.get("schema_info"):
            tables = list(state["schema_info"].get("tables", {}).keys())
            context_parts.append(f"Available tables: {', '.join(tables)}")
        
        if state.get("dataset_info"):
            context_parts.append(f"Dataset class: {state['dataset_info'].get('class_name')}")
            context_parts.append(f"Dataset file: {state['dataset_info'].get('file_path')}")
        
        if state.get("task_info"):
            context_parts.append(f"Task class: {state['task_info'].get('class_name')}")
            context_parts.append(f"Task type: {state['task_info'].get('task_type')}")
        
        return "\n".join(context_parts)
    
    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """
        Process agent result and extract state updates.
        
        Override in subclasses to handle specific outputs.
        """
        messages = result.get("messages", [])
        
        new_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                new_messages.append({
                    "role": "assistant",
                    "content": extract_text_content(msg.content),
                    "timestamp": None,
                })
        
        return {"messages": new_messages}
