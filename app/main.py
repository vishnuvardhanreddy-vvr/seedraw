"""
main.py

This FastAPI application provides endpoints for dynamic database connection,
schema inspection, and synthetic data generation using large language models (LLMs).

It supports two modes of database connectivity:
1. **Direct PostgreSQL connection** via asyncpg.
2. **Remote connection through MCP (Model Control Protocol)** without sharing credentials.

The API also:
- Dynamically serves frontend HTML files.
- Retrieves full database schema including PK/FK relationships.
- Uses a language model (e.g., Gemini 2.5 Flash) to generate and insert realistic sample data
  into the connected database.

Typical Usage Flow:
-------------------
1. Connect to a database:
    POST /connect-db
2. Retrieve schema:
    GET /get-schema
3. Generate synthetic data:
    POST /generate-data

Author: <Your Name / Org>
Version: 1.0.0
"""
import logging
import json
import os
from datetime import datetime, date
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.db_adapter import DBAdapter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="AI-Powered DB Schema and Data Generator")

# Serve static UI files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Global database adapter instance
db_adapter = DBAdapter()

# Default HTML page
current_html_file = "normalUI.html"


@app.get("/set_ui")
async def set_ui(filename: str):
    """
    Switch which HTML UI file is served at the root endpoint (`/`).

    Args:
        filename (str): Name of the HTML file (must exist under `/static`).

    Returns:
        dict: Confirmation message or error if file not found.
    """
    global current_html_file

    file_path = f"static/{filename}"
    if not os.path.exists(file_path):
        logger.warning("UI file not found: %s", file_path)
        return {"error": f"File '{filename}' not found in static/"}

    current_html_file = filename
    logger.info("UI file changed to %s", filename)
    return {"message": f"UI file changed to '{filename}'"}


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Serve the currently active HTML UI file at the root endpoint.

    Args:
        request (Request): FastAPI request object.

    Returns:
        HTMLResponse: Rendered HTML page.
    """
    return templates.TemplateResponse(current_html_file, {"request": request})


class DBConnectionRequest(BaseModel):
    """Pydantic model for database connection request."""

    host: str
    port: int
    user: str
    password: str
    database: str
    mcp_url: Optional[str] = Field(
        None,
        description="MCP server URL for credential-free connection mode."
    )


@app.post("/connect-db")
async def connect_db(req: DBConnectionRequest):
    """
    Establish a database connection, either via direct PostgreSQL credentials
    or using an MCP server URL.

    Args:
        req (DBConnectionRequest): Database connection parameters.

    Returns:
        dict: Connection status and mode.

    Raises:
        HTTPException: If connection fails or credentials are missing.
    """
    try:
        await db_adapter.close()  # Close any previous connection

        if req.mcp_url:
            await db_adapter.connect_mcp(req.mcp_url)
            mode = "mcp"
        else:
            if not all([req.host, req.port, req.user, req.password, req.database]):
                raise HTTPException(status_code=400, detail="Missing database credentials")

            await db_adapter.connect_direct(
                host=req.host,
                port=req.port,
                user=req.user,
                password=req.password,
                database=req.database
            )
            mode = "direct"

        logger.info("Database connection established in %s mode.", mode)
        return {"status": "success", "mode": mode}

    except Exception as e:
        logger.exception("Database connection failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


@app.get("/get-schema")
async def get_schema():
    """
    Retrieve the schema of the currently connected database, including:
    - Table names
    - Column names, data types, max lengths
    - Primary key and foreign key relationships

    Returns:
        dict: Database schema representation.

    Raises:
        HTTPException: If schema retrieval or query execution fails.
    """
    try:
        # --- Fetch columns ---
        column_rows = await db_adapter.query("""
            SELECT table_name, column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_name, ordinal_position;
        """)
        if isinstance(column_rows, list):
            column_rows = [json.loads(r) for r in column_rows]

        # --- Fetch foreign keys ---
        fk_rows = await db_adapter.query("""
            SELECT
                tc.table_name AS table_name,
                kcu.column_name AS column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY';
        """)
        if isinstance(fk_rows, list):
            fk_rows = [json.loads(r) for r in fk_rows]

        # --- Fetch primary keys ---
        pk_rows = await db_adapter.query("""
            SELECT
                tc.table_name,
                kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY';
        """)
        if isinstance(pk_rows, list):
            pk_rows = [json.loads(r) for r in pk_rows]

        # --- Construct schema dict ---
        schema: Dict[str, List[Dict[str, Any]]] = {}
        for row in column_rows:
            table = row["table_name"]
            schema.setdefault(table, []).append({
                "column": row["column_name"],
                "data_type": row["data_type"],
                "max_length": row["character_maximum_length"],
                "is_primary_key": False,
                "is_foreign_key": False,
                "references": None
            })

        # Flag PKs
        for pk in pk_rows:
            for col in schema.get(pk["table_name"], []):
                if col["column"] == pk["column_name"]:
                    col["is_primary_key"] = True

        # Flag FKs
        for fk in fk_rows:
            for col in schema.get(fk["table_name"], []):
                if col["column"] == fk["column_name"]:
                    col["is_foreign_key"] = True
                    col["references"] = {
                        "table": fk["foreign_table_name"],
                        "column": fk["foreign_column_name"]
                    }

        logger.info("Schema successfully fetched with %d tables.", len(schema))
        return {"status": "success", "schema": schema}

    except Exception as e:
        logger.exception("Failed to fetch database schema: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch schema: {str(e)}")



class GenerateDataRequest(BaseModel):
    """Request model for synthetic data generation."""

    batch_size: int = 1
    row_count: int = 1
    table_schema: Dict[str, List[Dict[str, Any]]]


@app.post("/generate-data")
async def generate_data(request: GenerateDataRequest = Body(...)):
    """
    Generate and insert synthetic data into the connected database.

    The function:
    - Builds a dependency graph to respect foreign key constraints.
    - Uses an LLM (e.g., Gemini 2.5 Flash) to produce realistic JSON data.
    - Inserts generated data into the database while respecting type/length limits.

    Args:
        request (GenerateDataRequest): Schema and parameters for data generation.

    Returns:
        dict: Summary of inserted rows, presence of foreign keys, and generated data.

    Raises:
        HTTPException: For LLM generation issues or database errors.
    """
    batch_size = request.batch_size
    schema = request.table_schema
    rows_per_table = request.row_count

    # --- Initialize LLM ---
    try:
        llm = init_chat_model(
            model="gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0.9,
        )
    except Exception as e:
        logger.exception("LLM initialization failed: %s", e)
        raise HTTPException(status_code=500, detail=f"LLM init failed: {str(e)}")

    # --- Detect foreign keys and build dependency graph ---
    has_foreign_keys = any(
        any(col.get("is_foreign_key") and col.get("references") for col in columns)
        for columns in schema.values()
    )

    if has_foreign_keys:
        graph = defaultdict(list)
        all_tables = set(schema.keys())
        for table_name, columns in schema.items():
            for col in columns:
                if col.get("is_foreign_key") and col.get("references"):
                    ref_table = col["references"]["table"]
                    if ref_table not in all_tables:
                        raise HTTPException(status_code=400,
                        detail=f"Foreign key references unknown table {ref_table}")
                    graph[ref_table].append(table_name)

        indegree = {table: 0 for table in all_tables}
        for deps in graph.values():
            for dep in deps:
                indegree[dep] += 1

        queue = deque([t for t in all_tables if indegree[t] == 0])
        sorted_tables = []

        while queue:
            t = queue.popleft()
            sorted_tables.append(t)
            for neighbor in graph[t]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_tables) != len(all_tables):
            raise HTTPException(status_code=400, detail="Circular foreign key dependency detected")
    else:
        sorted_tables = list(schema.keys())

    # --- Build LLM prompt ---
    seed = os.urandom(4).hex()
    prompt = f"""
        You are a JSON-based synthetic data generator. Seed: {seed}
        Generate {rows_per_table} rows of realistic data for each table listed below.
        Each table must appear as a key in the final JSON object.
        Output strictly valid JSON only (no markdown, no commentary).
    """

    if has_foreign_keys:
        prompt += "- Maintain foreign key consistency between tables.\n"
    else:
        prompt += "- Tables are independent; generate unrelated data.\n"

    for table_name in sorted_tables:
        prompt += f"\nTable: {table_name}\n"
        for col in schema[table_name]:
            col_line = f"- {col['column']}: {col['data_type']}"
            if col.get("max_length"):
                col_line += f"({col['max_length']})"
            if col.get("is_primary_key"):
                col_line += " [PRIMARY KEY]"
            if col.get("is_foreign_key") and col.get("references"):
                ref = col["references"]
                col_line += f" [FOREIGN KEY → {ref['table']}.{ref['column']}]"
            prompt += col_line + "\n"

    # --- Generate data via LLM ---
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt.strip())])
        text = response.content.strip()
        if "```json" in text:
            text = text.replace("```json", "").replace("```", "")
        all_data = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Invalid JSON returned from model: %s", text)
        raise HTTPException(status_code=500, detail=f"Invalid JSON returned from model: {text}")
    except Exception as e:
        logger.exception("Model error: %s", e)
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    # --- Validate model output ---
    missing_tables = [t for t in sorted_tables if t not in all_data]
    if missing_tables:
        raise HTTPException(status_code=500,
                            detail=f"Model output missing tables: {missing_tables}")

    # --- Insert generated data into DB ---
    try:
        for table in sorted_tables:
            rows = all_data.get(table, [])
            if not rows:
                continue

            col_meta = await db_adapter.query(f"""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = '{table}'
            """)
            col_meta = [json.loads(r) for r in col_meta]
            col_limits = {r["column_name"]: r["character_maximum_length"] for r in col_meta}
            cols = list(rows[0].keys())

            for row in rows:
                sql_values = []
                for col in cols:
                    val = row.get(col)
                    limit = col_limits.get(col)

                    if limit and isinstance(val, str) and len(val) > limit:
                        val = val[:limit]

                    col_type = next(
                        (r.get("data_type", "") for r in col_meta if r.get("column_name") == col), ""
                    ).lower()

                    # Handle type conversions
                    if "timestamp" in col_type and isinstance(val, str):
                        try:
                            iso_val = val.replace("Z", "").strip()
                            if "T" not in iso_val:
                                iso_val += "T00:00:00"
                            val = datetime.fromisoformat(iso_val)
                        except Exception:
                            val = datetime.utcnow()
                        val_str = f"'{val.strftime('%Y-%m-%dT%H:%M:%S')}'"
                    elif "date" in col_type and isinstance(val, str):
                        try:
                            val = date.fromisoformat(val.split("T")[0])
                        except Exception:
                            val = date.today()
                        val_str = f"'{val.isoformat()}'"
                    elif isinstance(val, str):
                        val_str = f"'{val.replace('\'', '\'\'')}'"
                    elif val is None:
                        val_str = "NULL"
                    else:
                        val_str = str(val)

                    sql_values.append(val_str)

                col_names = ', '.join(cols)
                sql_query = f"INSERT INTO {table} ({col_names}) VALUES ({', '.join(sql_values)})"
                await db_adapter.execute(sql_query)

    except Exception as e:
        logger.exception("Database insertion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

    logger.info("Data generation and insertion successful.")
    return {
        "status": "success",
        "foreign_keys_present": has_foreign_keys,
        "rows_inserted": sum(len(v) for v in all_data.values()),
        "generated_data": all_data,
    }
