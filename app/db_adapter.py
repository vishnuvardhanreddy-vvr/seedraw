"""
db_adapter.py

This module defines the `DBAdapter` class, a unified asynchronous interface for 
connecting to and querying a PostgreSQL database either directly via `asyncpg` 
or through a remote MCP (Model Control Protocol) server.

The adapter automatically handles connection pooling, switching between 
connection modes, and executing SQL queries in a consistent way.

Usage Example:
    adapter = DBAdapter()
    await adapter.connect_direct(
        host="localhost", port=5432, user="user", password="pass", database="mydb"
    )
    rows = await adapter.query("SELECT * FROM users;")
    await adapter.execute("INSERT INTO users (name) VALUES ($1);", ["Alice"])
    await adapter.close()

    # OR using MCP mode
    await adapter.connect_mcp("https://mcp-server.example.com")
    rows = await adapter.query("SELECT * FROM remote_table;")
    await adapter.close()
"""
import logging
from typing import Any, Dict, List, Optional
import asyncpg
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


class DBAdapter:
    """Database Adapter supporting both direct PostgreSQL and MCP-based connections."""

    def __init__(self) -> None:
        """
        Initialize the DBAdapter instance.

        Attributes:
            pg_pool (Optional[asyncpg.pool.Pool]): Connection pool for PostgreSQL.
            mcp_tools (Optional[Dict[str, Any]]): Registered MCP tools if connected via MCP.
            mode (str): Current connection mode ("none", "direct", or "mcp").
        """
        self.pg_pool: Optional[asyncpg.pool.Pool] = None
        self.mcp_tools: Optional[Dict[str, Any]] = None
        self.mode: str = "none"

    async def connect_direct(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ) -> None:
        """
        Establish a direct asynchronous connection to PostgreSQL using asyncpg.

        Args:
            host (str): Database host.
            port (int): Database port.
            user (str): Username for authentication.
            password (str): User password.
            database (str): Target database name.

        Raises:
            asyncpg.PostgresError: If the connection cannot be established.
        """
        if self.pg_pool:
            logger.info("Closing existing PostgreSQL connection pool.")
            await self.pg_pool.close()

        try:
            self.pg_pool = await asyncpg.create_pool(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                min_size=1,
                max_size=5,
            )
            self.mode = "direct"
            logger.info("Connected to PostgreSQL database at %s:%d", host, port)
        except Exception as e:
            logger.exception("Failed to establish PostgreSQL connection: %s", e)
            raise

    async def connect_mcp(self, mcp_url: str) -> None:
        """
        Connect to a database through a Model Control Protocol (MCP) server.

        Args:
            mcp_url (str): URL of the MCP server.

        Raises:
            Exception: If the MCP server does not provide required tools ('query', 'execute').
        """
        logger.info("Connecting to MCP server at %s", mcp_url)
        mcp_client = MultiServerMCPClient(
            {
                "DBTools": {"url": mcp_url, "transport": "streamable_http"},
            }
        )
        tools_list = await mcp_client.get_tools()
        self.mcp_tools = {tool.name: tool for tool in tools_list}

        if "query" not in self.mcp_tools or "execute" not in self.mcp_tools:
            logger.error("MCP server is missing required tools: 'query' and/or 'execute'.")
            raise Exception("MCP server must provide 'query' and 'execute' tools")

        self.mode = "mcp"
        logger.info("Connected to MCP server with %d tools.", len(self.mcp_tools))

    async def query(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL SELECT query and return the result.

        Args:
            sql (str): The SQL SELECT statement to execute.

        Returns:
            List[Dict[str, Any]]: List of result rows as dictionaries.

        Raises:
            Exception: If not connected to any database.
        """

        if self.mode == "mcp":
            return await self.mcp_tools["query"].ainvoke({"sql": sql})
        elif self.mode == "direct":
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(sql)
                return [dict(r) for r in rows]
        else:
            logger.error("Query attempted without an active connection.")
            raise Exception("Database not connected")

    async def execute(self, sql: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Execute a SQL command (INSERT, UPDATE, DELETE, etc.).

        Args:
            sql (str): The SQL statement to execute.
            params (Optional[List[Any]]): List of parameters to bind to the SQL.

        Returns:
            Dict[str, Any]: Execution result or confirmation message.

        Raises:
            Exception: If not connected to any database.
        """
        logger.debug("Executing SQL command: %s | Params: %s", sql, params)

        if self.mode == "mcp":
            return await self.mcp_tools["execute"].ainvoke({
                "input": {
                    "sql": sql,
                    "params": params or []
                }
            })
        elif self.mode == "direct":
            async with self.pg_pool.acquire() as conn:
                await conn.execute(sql, *(params or []))
            return {"status": "ok"}
        else:
            logger.error("Execution attempted without an active connection.")
            raise Exception("Database not connected")

    async def close(self) -> None:
        """
        Close all active connections and reset adapter state.

        Ensures that any open PostgreSQL pools are closed and that the adapter
        returns to its uninitialized state.
        """
        if self.pg_pool:
            logger.info("Closing PostgreSQL connection pool.")
            await self.pg_pool.close()

        self.pg_pool = None
        self.mcp_tools = None
        self.mode = "none"
        logger.info("DBAdapter has been closed and reset.")
