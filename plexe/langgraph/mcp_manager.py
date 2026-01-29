import logging
import json
import os
import re
import sys
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import Tool, BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv
from contextlib import AsyncExitStack

logger = logging.getLogger(__name__)

class MCPManager:
    """
    Manager for Model Context Protocol (MCP) servers.
    Provides connectivity and conversion of MCP tools to LangChain tools.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get("MCP_CONFIG_PATH", "mcp_config.json")
        self.sessions: Dict[str, Any] = {}
        self.tools: List[BaseTool] = []
        self._exit_stack = AsyncExitStack()
        # Load environment variables from .env
        load_dotenv()
        
    async def initialize(self):
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

    async def _connect_to_server(self, name: str, config: Dict[str, Any]):
        """Connect to a specific MCP server and discover tools."""
        try:
            command = config.get("command")
            args = config.get("args", [])
            
            # If command is "python", use the current Python interpreter
            # This ensures we use the venv Python, not system Python
            if command == "python":
                command = sys.executable
            
            # Expand environment variables in env config
            env_config = config.get("env", {})
            env = {**os.environ}
            for k, v in env_config.items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    var_name = v[2:-1]
                    env[k] = os.environ.get(var_name, "")
                    logger.debug(f"Expanded MCP env var {k}={var_name}")
                else:
                    env[k] = v
            
            # Convert relative paths to absolute
            abs_args = []
            for arg in args:
                if arg.endswith('.py') and not os.path.isabs(arg):
                    abs_args.append(os.path.abspath(arg))
                else:
                    abs_args.append(arg)
            
            params = StdioServerParameters(command=command, args=abs_args, env=env)
            
            # Use proper async context manager pattern
            logger.info(f"Connecting to MCP server: {name}...")
            
            client_ctx = stdio_client(params)
            streams = await self._exit_stack.enter_async_context(client_ctx)
            read_stream, write_stream = streams
            
            session = ClientSession(read_stream, write_stream)
            await self._exit_stack.enter_async_context(session)
            await session.initialize()
            
            # List tools from the server
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
        import asyncio
        
        async def tool_func_async(**kwargs):
            result = await session.call_tool(tool_info.name, kwargs)
            # Handle list of content parts from MCP
            return "\n".join([str(c.text) if hasattr(c, 'text') else str(c) for c in result.content])

        def tool_func_sync(**kwargs):
            try:
                # Try to get existing loop or create new one
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We are in an async world already, this is tricky
                        # But for now, we'll use a wrapper if needed
                        import nest_asyncio
                        nest_asyncio.apply()
                        return loop.run_until_complete(tool_func_async(**kwargs))
                    else:
                        return loop.run_until_complete(tool_func_async(**kwargs))
                except RuntimeError:
                    return asyncio.run(tool_func_async(**kwargs))
            except Exception as e:
                return f"MCP tool error: {str(e)}"

        # Check if tool has input schema with multiple properties
        input_schema = getattr(tool_info, 'inputSchema', None)
        
        if input_schema and isinstance(input_schema, dict):
            properties = input_schema.get('properties', {})
            required = input_schema.get('required', [])
            
            if properties:
                # Build Pydantic model dynamically for StructuredTool
                field_definitions = {}
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get('type', 'string')
                    prop_desc = prop_info.get('description', '')
                    
                    # Map JSON schema types to Python types
                    type_mapping = {
                        'string': str,
                        'integer': int,
                        'number': float,
                        'boolean': bool,
                        'array': list,
                        'object': dict,
                    }
                    python_type = type_mapping.get(prop_type, str)
                    
                    # Special handling for array types: ensure items schema is properly specified
                    if prop_type == 'array':
                        items_schema = prop_info.get('items')
                        if items_schema is None or not isinstance(items_schema, dict):
                            # If items is missing or invalid, default to list of dicts
                            # This ensures Gemini gets a valid schema
                            prop_info['items'] = {'type': 'object'}
                            logger.warning(f"Array parameter '{prop_name}' missing 'items' schema, defaulting to 'object'")
                        elif 'type' not in items_schema:
                            # If items exists but lacks 'type', add it
                            items_schema['type'] = 'object'
                            logger.warning(f"Array parameter '{prop_name}' items missing 'type', defaulting to 'object'")
                    
                    # Use Optional for non-required fields
                    if prop_name in required:
                        field_definitions[prop_name] = (python_type, Field(description=prop_desc))
                    else:
                        field_definitions[prop_name] = (Optional[python_type], Field(default=None, description=prop_desc))
                
                # Create dynamic Pydantic model for args schema
                ArgsModel = create_model(f'{tool_info.name}Args', **field_definitions)
                
                return StructuredTool.from_function(
                    func=tool_func_sync,
                    name=tool_info.name,
                    description=tool_info.description or f"MCP tool: {tool_info.name}",
                    args_schema=ArgsModel,
                )
        
        # Fallback to simple Tool for tools without complex schema
        return Tool(
            name=tool_info.name,
            description=tool_info.description or f"MCP tool: {tool_info.name}",
            func=tool_func_sync,
        )

    def get_tools(self) -> List[BaseTool]:
        """Return the list of discovered MCP tools."""
        return self.tools

    async def close(self):
        """Close all MCP server connections."""
        try:
            await self._exit_stack.aclose()
            self.sessions.clear()
            self.tools.clear()
            logger.info("MCP Manager closed all connections")
        except Exception as e:
            logger.error(f"Error closing MCP Manager: {e}")