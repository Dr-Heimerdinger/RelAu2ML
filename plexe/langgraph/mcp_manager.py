import logging
import json
import os
import re
import sys
import time
import asyncio
import threading
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import Tool, BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv
from contextlib import AsyncExitStack

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Prevents repeated calls to a failing service.

    After *failure_threshold* consecutive failures the circuit opens and
    rejects calls immediately for *reset_timeout* seconds, then auto-resets.
    """

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 300.0):
        self._failures: Dict[str, int] = {}
        self._open_until: Dict[str, float] = {}
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

    def is_open(self, key: str) -> bool:
        deadline = self._open_until.get(key)
        if deadline is None:
            return False
        if time.monotonic() < deadline:
            return True
        del self._open_until[key]
        self._failures[key] = 0
        return False

    def record_failure(self, key: str):
        self._failures[key] = self._failures.get(key, 0) + 1
        if self._failures[key] >= self.failure_threshold:
            self._open_until[key] = time.monotonic() + self.reset_timeout
            logger.warning(f"Circuit breaker opened for '{key}' after {self._failures[key]} failures")

    def record_success(self, key: str):
        self._failures.pop(key, None)
        self._open_until.pop(key, None)


class MCPManager:
    """
    Manager for Model Context Protocol (MCP) servers.

    Uses a persistent background thread with a long-lived event loop so that
    MCP sessions (and their stdio transports) remain valid for the entire
    lifetime of the manager.  Both initialization and tool invocations run
    on this same loop via ``asyncio.run_coroutine_threadsafe``.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get("MCP_CONFIG_PATH", "mcp_config.json")
        self.sessions: Dict[str, Any] = {}
        self.tools: List[BaseTool] = []
        self._exit_stack = AsyncExitStack()
        self.circuit_breaker = CircuitBreaker()
        load_dotenv()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._start_background_loop()

    def _start_background_loop(self):
        """Start a daemon thread that runs an asyncio event loop forever."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mcp-event-loop"
        )
        self._thread.start()

    def _run_loop(self):
        """Entry point for the background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coroutine(self, coro, timeout: float = 60.0):
        """Submit a coroutine to the persistent loop and wait for the result."""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("MCPManager background event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def initialize_sync(self, timeout: float = 30.0):
        """Synchronous entry point: initialize all MCP server connections."""
        self._run_coroutine(self._initialize(), timeout=timeout)

    async def _initialize(self):
        """Initialize connections to configured MCP servers."""
        if not os.path.exists(self.config_path):
            logger.warning(f"MCP config not found at {self.config_path}")
            return

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            for server_name, server_config in config.get("mcpServers", {}).items():
                await self._connect_to_server(server_name, server_config)

        except Exception as e:
            logger.error(f"Error initializing MCP Manager: {e}")

    async def initialize(self):
        """Async entry point (kept for backward compatibility)."""
        await self._initialize()

    async def _connect_to_server(self, name: str, config: Dict[str, Any]):
        """Connect to a specific MCP server and discover tools."""
        try:
            command = config.get("command")
            args = config.get("args", [])

            if command == "python":
                command = sys.executable

            env_config = config.get("env", {})
            env = {**os.environ}
            for k, v in env_config.items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    var_name = v[2:-1]
                    env[k] = os.environ.get(var_name, "")
                    logger.debug(f"Expanded MCP env var {k}={var_name}")
                else:
                    env[k] = v

            abs_args = []
            for arg in args:
                if arg.endswith('.py') and not os.path.isabs(arg):
                    abs_args.append(os.path.abspath(arg))
                else:
                    abs_args.append(arg)

            params = StdioServerParameters(command=command, args=abs_args, env=env)

            logger.info(f"Connecting to MCP server: {name}...")

            client_ctx = stdio_client(params)
            streams = await self._exit_stack.enter_async_context(client_ctx)
            read_stream, write_stream = streams

            session = ClientSession(read_stream, write_stream)
            await self._exit_stack.enter_async_context(session)
            await session.initialize()

            result = await session.list_tools()

            for tool_info in result.tools:
                langchain_tool = self._convert_to_langchain_tool(session, tool_info)
                self.tools.append(langchain_tool)

            self.sessions[name] = session
            logger.info(f"Connected to MCP server: {name} with {len(result.tools)} tools")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {name}: {e}")

    def _convert_to_langchain_tool(self, session: ClientSession, tool_info: Any) -> BaseTool:
        """Convert an MCP tool definition to a LangChain BaseTool with proper schema."""

        persistent_loop = self._loop
        cb = self.circuit_breaker
        tool_name = tool_info.name

        async def tool_func_async(**kwargs):
            # Filter out None values — MCP servers define their own defaults
            # for optional params.  LLMs sometimes pass None explicitly which
            # fails Pydantic validation on the server side.
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            result = await session.call_tool(tool_name, filtered_kwargs)
            return "\n".join([str(c.text) if hasattr(c, 'text') else str(c) for c in result.content])

        def tool_func_sync(**kwargs):
            if cb.is_open(tool_name):
                return f"MCP tool '{tool_name}' temporarily unavailable (circuit breaker open). Skipping."
            try:
                if persistent_loop is None or persistent_loop.is_closed():
                    return "MCP tool error: background event loop is not running"
                future = asyncio.run_coroutine_threadsafe(
                    tool_func_async(**kwargs), persistent_loop
                )
                result = future.result(timeout=120)
                cb.record_success(tool_name)
                return result
            except Exception as e:
                cb.record_failure(tool_name)
                return f"MCP tool error: {str(e)}"

        input_schema = getattr(tool_info, 'inputSchema', None)

        if input_schema and isinstance(input_schema, dict):
            properties = input_schema.get('properties', {})
            required = input_schema.get('required', [])

            if properties:
                field_definitions = {}
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get('type', 'string')
                    prop_desc = prop_info.get('description', '')

                    type_mapping = {
                        'string': str,
                        'integer': int,
                        'number': float,
                        'boolean': bool,
                        'array': list,
                        'object': dict,
                    }
                    python_type = type_mapping.get(prop_type, str)

                    if prop_type == 'array':
                        items_schema = prop_info.get('items')
                        if items_schema is None or not isinstance(items_schema, dict):
                            prop_info['items'] = {'type': 'object'}
                            logger.warning(f"Array parameter '{prop_name}' missing 'items' schema, defaulting to 'object'")
                        elif 'type' not in items_schema:
                            items_schema['type'] = 'object'
                            logger.warning(f"Array parameter '{prop_name}' items missing 'type', defaulting to 'object'")

                    if prop_name in required:
                        field_definitions[prop_name] = (python_type, Field(description=prop_desc))
                    else:
                        field_definitions[prop_name] = (Optional[python_type], Field(default=None, description=prop_desc))

                ArgsModel = create_model(f'{tool_info.name}Args', **field_definitions)

                return StructuredTool.from_function(
                    func=tool_func_sync,
                    name=tool_info.name,
                    description=tool_info.description or f"MCP tool: {tool_info.name}",
                    args_schema=ArgsModel,
                )

        return Tool(
            name=tool_info.name,
            description=tool_info.description or f"MCP tool: {tool_info.name}",
            func=tool_func_sync,
        )

    def get_tools(self) -> List[BaseTool]:
        """Return the list of discovered MCP tools."""
        return self.tools

    async def _close(self):
        """Async close: tear down sessions and exit stack."""
        try:
            await self._exit_stack.aclose()
            self.sessions.clear()
            self.tools.clear()
            logger.info("MCP Manager closed all connections")
        except Exception as e:
            logger.error(f"Error closing MCP Manager: {e}")

    def close(self):
        """Close all MCP server connections and stop the background loop."""
        if self._loop and not self._loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(self._close(), self._loop)
                future.result(timeout=10)
            except Exception as e:
                logger.error(f"Error during MCP Manager close: {e}")
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=5)

    async def aclose(self):
        """Async-compatible close (delegates to sync close for cleanup)."""
        self.close()