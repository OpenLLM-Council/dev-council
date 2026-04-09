"""MCP client support for dev-council.

Usage
-----
MCP servers are configured in one of two JSON files:

  ~/.dev-council/mcp.json
  .mcp.json

Project config overrides user config by server name.
MCP tools are auto-registered as `mcp__<server>__<tool>`.
"""
from .types import MCPServerConfig, MCPTool, MCPServerState, MCPTransport  # noqa: F401
from .client import MCPClient, MCPManager, get_mcp_manager  # noqa: F401
from .config import (  # noqa: F401
    load_mcp_configs,
    save_user_mcp_config,
    add_server_to_user_config,
    remove_server_from_user_config,
    list_config_files,
)
from .tools import initialize_mcp, reload_mcp, refresh_server  # noqa: F401
