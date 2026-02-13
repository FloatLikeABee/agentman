"""
Text-to-SQL submodule (database module).

Workflow:
1. Text-to-SQL: LLM converts natural language to SQL using schema.
2. Retrieval: Execute the query and get rows.
3. Augmentation: LLM formats/summarizes the results.

Supports existing db-tool profile (db_tool_id), connection_config, connection_string (OLE DB),
or default SQL Server connection from config (see config.text_to_sql_default_*).
Schema can be provided as schema_text or introspected from schema_tables.
"""
import logging
import re
from typing import Dict, Any, List, Optional

from .config import settings
from .models import (
    DatabaseConnectionConfig,
    DatabaseToolProfile,
    DatabaseType,
    LLMProviderType,
)
from .llm_factory import LLMFactory, LLMProvider


# Default SQL Server port
DEFAULT_SQL_SERVER_PORT = 1433


def _parse_connection_string(connection_string: str) -> Dict[str, str]:
    """Parse a connection string into key-value pairs (keys lowercased). Handles Key=Value; and Key=Value."""
    if not connection_string or not connection_string.strip():
        return {}
    out = {}
    for part in connection_string.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        k = k.strip().lower()
        v = v.strip()
        if k and v is not None:
            out[k] = v
    return out


def _normalize_oledb_connection_string(connection_string: str) -> str:
    """Extract essential info from any connection string and build a canonical OLE DB string for SQL Server."""
    parsed = _parse_connection_string(connection_string)
    if not parsed:
        return connection_string
    # Map various key names to canonical values
    provider = (
        parsed.get("provider")
        or parsed.get("driver")
    ) or "SQLOLEDB"
    data_source = parsed.get("data source") or parsed.get("server") or parsed.get("host") or ""
    initial_catalog = parsed.get("initial catalog") or parsed.get("database") or ""
    user_id = parsed.get("user id") or parsed.get("uid") or parsed.get("username") or ""
    password = parsed.get("password") or parsed.get("pwd") or ""
    # Build canonical string (order and key names that SQL Server OLE DB expects)
    parts = [
        f"Provider={provider};",
        f"Data Source={data_source};",
        f"Initial Catalog={initial_catalog};",
        f"User ID={user_id};",
        f"Password={password};",
    ]
    return "".join(parts)


def _build_oledb_connection_string(config: DatabaseConnectionConfig) -> str:
    """Build an OLE DB (SQLOLEDB) connection string from connection_config for fallback when ODBC is unavailable."""
    port = getattr(config, "port", None) or DEFAULT_SQL_SERVER_PORT
    password = (config.password or "").strip() or getattr(settings, "text_to_sql_default_password", None) or ""
    if not password:
        password = "$transfinder2006"
    return (
        f"Provider=SQLOLEDB;"
        f"Data Source={config.host},{port};"
        f"Initial Catalog={config.database};"
        f"User ID={config.username};"
        f"Password={password};"
    )


def _get_llm_caller(provider_str: str, model_name: Optional[str]) -> Any:
    """Resolve LLM provider and model from settings."""
    provider_str = (provider_str or "qwen").lower().strip()
    if provider_str == "gemini":
        api_key = settings.gemini_api_key
        model = model_name or settings.gemini_default_model
        provider = LLMProvider.GEMINI
    elif provider_str == "qwen":
        api_key = settings.qwen_api_key
        model = model_name or settings.qwen_default_model
        provider = LLMProvider.QWEN
    elif provider_str == "mistral":
        api_key = settings.mistral_api_key
        model = model_name or settings.mistral_default_model
        provider = LLMProvider.MISTRAL
    else:
        api_key = settings.qwen_api_key
        model = model_name or settings.qwen_default_model
        provider = LLMProvider.QWEN
    return LLMFactory.create_caller(provider=provider, api_key=api_key, model=model, temperature=0.1, max_tokens=4096)


def _run_sql_server_query(config: DatabaseConnectionConfig, sql: str) -> Dict[str, Any]:
    """Run a single SQL query against SQL Server using connection_config (no stored profile)."""
    try:
        import pyodbc
    except ImportError:
        raise ValueError("pyodbc is required for Text-to-SQL with connection_config. Install: pip install pyodbc")
    port = getattr(config, "port", None) or DEFAULT_SQL_SERVER_PORT
    drivers_to_try = [
        "ODBC Driver 17 for SQL Server",
        "ODBC Driver 18 for SQL Server",
        "SQL Server",
        "SQL Server Native Client 11.0",
    ]
    connection = None
    last_error = None
    for driver in drivers_to_try:
        try:
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={config.host},{port};"
                f"DATABASE={config.database};"
                f"UID={config.username};"
                f"PWD={config.password}"
            )
            connection = pyodbc.connect(conn_str, timeout=30)
            break
        except Exception as e:
            last_error = e
            continue
    if not connection:
        raise ValueError(f"Failed to connect to SQL Server. Last error: {last_error}")
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        columns = [column[0] for column in cursor.description] if cursor.description else []
        rows = [list(row) for row in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "total_rows": len(rows)}
    finally:
        connection.close()


def _run_sql_via_oledb(connection_string: str, sql: str) -> Dict[str, Any]:
    """Run a single SQL query using OLE DB. Uses essential info from the connection string (any format) to build a canonical SQL Server OLE DB string."""
    try:
        import adodbapi
    except ImportError:
        raise ValueError(
            "adodbapi is required for Text-to-SQL with connection_string (OLE DB). Install: pip install adodbapi. On Windows, pywin32 may also be required."
        )
    canonical = _normalize_oledb_connection_string(connection_string)
    conn = None
    try:
        conn = adodbapi.connect(canonical)
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [column[0] for column in cursor.description] if cursor.description else []
        rows = [list(row) for row in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "total_rows": len(rows)}
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _get_all_tables_sqlserver(config: DatabaseConnectionConfig) -> List[str]:
    """Return list of user table names for SQL Server (for optional schema introspection)."""
    sql = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME"
    result = _run_sql_server_query(config, sql)
    return [row[0] for row in result.get("rows", []) if row and row[0]]


def _get_all_tables_sqlserver_oledb(connection_string: str) -> List[str]:
    """Return list of user table names for SQL Server via OLE DB."""
    sql = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME"
    result = _run_sql_via_oledb(connection_string, sql)
    return [row[0] for row in result.get("rows", []) if row and row[0]]


def _introspect_schema_sqlserver(config: DatabaseConnectionConfig, table_names: List[str]) -> str:
    """Introspect SQL Server schema for given tables and return a string description for the LLM."""
    if not table_names:
        return ""
    quoted = ", ".join(f"'{t}'" for t in table_names)
    sql = f"""
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME IN ({quoted})
    ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    result = _run_sql_server_query(config, sql)
    lines = []
    current_table = None
    for row in result.get("rows", []):
        table_name, column_name, data_type, is_nullable, max_len = row[0], row[1], row[2], row[3], row[4]
        if table_name != current_table:
            current_table = table_name
            lines.append(f"\nTable: {table_name}")
        part = f"  - {column_name} ({data_type}"
        if max_len:
            part += f", max_length={max_len}"
        part += ", nullable=" + str(is_nullable) + ")"
        lines.append(part)
    return "\n".join(lines).strip() if lines else ""


def _introspect_schema_sqlserver_oledb(connection_string: str, table_names: List[str]) -> str:
    """Introspect SQL Server schema via OLE DB for given tables."""
    if not table_names:
        return ""
    quoted = ", ".join(f"'{t}'" for t in table_names)
    sql = f"""
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME IN ({quoted})
    ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    result = _run_sql_via_oledb(connection_string, sql)
    lines = []
    current_table = None
    for row in result.get("rows", []):
        table_name, column_name, data_type, is_nullable, max_len = row[0], row[1], row[2], row[3], row[4]
        if table_name != current_table:
            current_table = table_name
            lines.append(f"\nTable: {table_name}")
        part = f"  - {column_name} ({data_type}"
        if max_len:
            part += f", max_length={max_len}"
        part += ", nullable=" + str(is_nullable) + ")"
        lines.append(part)
    return "\n".join(lines).strip() if lines else ""


def _extract_sql_from_llm_response(text: str) -> str:
    """Extract SQL from LLM response (remove markdown code blocks and extra text)."""
    text = (text or "").strip()
    # Remove ```sql ... ``` or ``` ... ```
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # If no backticks, take first line that looks like SELECT/WITH
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
            return text  # return full text in case multi-line
    return text


class TextToSQLService:
    """Text-to-SQL workflow: natural language → SQL → run → summarize."""

    def __init__(self, db_tools_manager: Any):
        self.logger = logging.getLogger(__name__)
        self.db_tools_manager = db_tools_manager

    def _get_schema(
        self,
        schema_text: Optional[str],
        schema_tables: Optional[List[str]],
        profile: Optional[DatabaseToolProfile],
        connection_config: Optional[DatabaseConnectionConfig],
        connection_string: Optional[str] = None,
    ) -> str:
        if schema_text and schema_text.strip():
            return schema_text.strip()
        # Resolve table list: use provided list or introspect all tables (SQL Server)
        tables_to_use: List[str] = list(schema_tables) if schema_tables else []
        if not tables_to_use:
            if profile:
                if profile.db_type != DatabaseType.SQLSERVER:
                    raise ValueError("Optional schema (no table names) is supported for SQL Server only. Use schema_text or schema_tables for other DBs.")
                tables_to_use = _get_all_tables_sqlserver(profile.connection_config)
            elif connection_string:
                tables_to_use = _get_all_tables_sqlserver_oledb(connection_string)
            elif connection_config:
                tables_to_use = _get_all_tables_sqlserver(connection_config)
            if not tables_to_use:
                return "No tables found in the database. The user question may not be answerable with SQL."
        if profile:
            if profile.db_type != DatabaseType.SQLSERVER:
                raise ValueError("Schema introspection is supported for SQL Server only. Use schema_text for other DBs.")
            return _introspect_schema_sqlserver(profile.connection_config, tables_to_use)
        if connection_string:
            return _introspect_schema_sqlserver_oledb(connection_string, tables_to_use)
        if connection_config:
            return _introspect_schema_sqlserver(connection_config, tables_to_use)
        raise ValueError("Provide db_tool_id, connection_config, or connection_string to introspect schema")

    def _execute_sql(
        self,
        profile: Optional[DatabaseToolProfile],
        connection_config: Optional[DatabaseConnectionConfig],
        connection_string: Optional[str],
        sql: str,
        tool_id: Optional[str],
    ) -> Dict[str, Any]:
        if profile and tool_id:
            return self.db_tools_manager.execute_raw_sql(tool_id, sql)
        if connection_string:
            return _run_sql_via_oledb(connection_string, sql)
        if connection_config:
            return _run_sql_server_query(connection_config, sql)
        raise ValueError("Provide db_tool_id, connection_config, or connection_string to execute SQL")

    def run(
        self,
        question: str,
        db_tool_id: Optional[str] = None,
        connection_config: Optional[DatabaseConnectionConfig] = None,
        connection_string: Optional[str] = None,
        schema_tables: Optional[List[str]] = None,
        schema_text: Optional[str] = None,
        provider: str = "qwen",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full Text-to-SQL workflow.
        Returns dict with keys: sql, columns, rows, total_rows, summary, error (if any).
        """
        profile = None
        if db_tool_id and self.db_tools_manager:
            profile = self.db_tools_manager.get_profile(db_tool_id)
            if not profile:
                return {"sql": "", "columns": [], "rows": [], "total_rows": 0, "summary": "", "error": f"Database tool '{db_tool_id}' not found"}
            if profile.db_type == DatabaseType.MONGODB:
                return {"sql": "", "columns": [], "rows": [], "total_rows": 0, "summary": "", "error": "Text-to-SQL does not support MongoDB"}

        effective_connection_string = connection_string
        if not effective_connection_string and connection_config and not profile:
            effective_connection_string = _build_oledb_connection_string(connection_config)
            connection_config = None
        try:
            schema_desc = self._get_schema(schema_text, schema_tables, profile, connection_config, effective_connection_string)
        except Exception as e:
            err_msg = str(e)
            if (
                not effective_connection_string
                and connection_config
                and ("Failed to connect" in err_msg or "IM002" in err_msg or "Data source" in err_msg or "ODBC" in err_msg)
            ):
                try:
                    effective_connection_string = _build_oledb_connection_string(connection_config)
                    schema_desc = self._get_schema(schema_text, schema_tables, profile, None, effective_connection_string)
                except Exception as e2:
                    self.logger.exception("Schema resolution failed (ODBC and OLE DB fallback)")
                    return {"sql": "", "columns": [], "rows": [], "total_rows": 0, "summary": "", "error": str(e2)}
            else:
                self.logger.exception("Schema resolution failed")
                return {"sql": "", "columns": [], "rows": [], "total_rows": 0, "summary": "", "error": str(e)}

        # Step 1: LLM generates SQL
        llm = _get_llm_caller(provider, model)
        prompt_sql = (
            "Here is the schema for the database:\n\n"
            f"{schema_desc}\n\n"
            f"The user asked: {question}\n\n"
            "Generate a single SQL query (SQL Server dialect) to answer this question. "
            "Return only the SQL query, no explanation or markdown."
        )
        try:
            sql_raw = llm.generate(prompt_sql)
            sql = _extract_sql_from_llm_response(sql_raw)
        except Exception as e:
            self.logger.exception("LLM SQL generation failed")
            return {"sql": "", "columns": [], "rows": [], "total_rows": 0, "summary": "", "error": f"SQL generation failed: {e}"}

        # Step 2: Execute query
        try:
            result = self._execute_sql(profile, connection_config, effective_connection_string, sql, db_tool_id)
            columns = result.get("columns", [])
            rows = result.get("rows", [])
            total_rows = result.get("total_rows", len(rows))
        except Exception as e:
            self.logger.exception("Query execution failed")
            return {"sql": sql, "columns": [], "rows": [], "total_rows": 0, "summary": "", "error": f"Query execution failed: {e}"}

        # Step 3: LLM formats result as markdown (data table + summary/abstraction)
        rows_preview = rows[:100]
        prompt_summary = (
            f"The user asked: {question}\n\n"
            f"Query results: columns = {columns}, total rows = {total_rows}.\n"
            "Raw rows (each line is one row, values comma-separated):\n"
        )
        for r in rows_preview:
            prompt_summary += ",".join(str(c) if c is not None else "" for c in r) + "\n"
        if len(rows) > 100:
            prompt_summary += f"... and {len(rows) - 100} more rows.\n"
        prompt_summary += (
            "\nOutput your response in **markdown** only. Include:\n"
            "1. A markdown table of the data (so the user sees the fetched data).\n"
            "2. A short summary or abstraction in one or two paragraphs.\n"
            "Do not include any text outside the markdown."
        )
        try:
            summary = llm.generate(prompt_summary)
        except Exception as e:
            self.logger.exception("LLM summary failed")
            summary = f"Results retrieved ({total_rows} rows) but summary failed: {e}"

        return {
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "total_rows": total_rows,
            "summary": summary,
            "error": None,
        }
