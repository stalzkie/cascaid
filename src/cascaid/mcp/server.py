"""Cascaid MCP Server (PRD 7): exposes cascade-risk querying as a direct tool call
for any MCP-speaking agent (Claude Code/Desktop, an internal agent workflow, Claude
Tag in Slack). Runs as a local stdio subprocess like any MCP server a client
launches directly -- not network-exposed, so no auth (see docs/... Phase 1 for why
the HTTP APIs need it and this doesn't):

    python -m cascaid.mcp.server --database-url ... [--store data/graph_store]
"""

from __future__ import annotations

import argparse

from mcp.server.mcpserver import MCPServer

from cascaid.mcp.tools import get_cascade_risk
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import init_db


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=str, default="data/graph_store")
    parser.add_argument("--database-url", type=str, required=True)
    return parser.parse_args(argv)


def build_server(args: argparse.Namespace) -> MCPServer:
    init_db(get_engine(args.database_url))
    session_factory = make_session_factory(args.database_url)
    store_dir = args.store
    server = MCPServer("cascaid")

    @server.tool()
    def cascade_risk(run_id: str) -> dict | None:
        """Current cascade risk for every node in the given pipeline run, as last
        computed by Model Serving. Returns null if no snapshot has been ingested
        for that run_id yet."""
        return get_cascade_risk(store_dir, session_factory, run_id)

    return server


def build_server_from_argv(argv: list[str] | None = None) -> MCPServer:
    return build_server(parse_args(argv))


def main():
    args = parse_args()
    build_server(args).run(transport="stdio")


if __name__ == "__main__":
    main()
