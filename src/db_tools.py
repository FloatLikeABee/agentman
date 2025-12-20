import logging
import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import threading

from tinydb import TinyDB, Query

from .config import settings
from .models import (
    DatabaseToolProfile,
    DatabaseToolCreateRequest,
    DatabaseToolUpdateRequest,
    DatabaseType,
    DatabaseConnectionConfig,
    DatabaseConnectionConfigUpdate,
)


class DatabaseToolsManager:
    """Manage database tool profiles with connection configs, SQL statements, and caching."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_tools: Dict[str, DatabaseToolProfile] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}  # Cache for query results
        self.cache_lock = threading.Lock()

        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "db_tools.json")
        self.cache_db_path = os.path.join(settings.data_directory, "db_tools_cache.json")
        self.db = TinyDB(self.db_path)
        self.cache_db = TinyDB(self.cache_db_path)
        self.query = Query()

        self._load_db_tools()
        self._cleanup_expired_cache()

    def _load_db_tools(self) -> None:
        """Load database tool profiles from TinyDB."""
        try:
            docs = self.db.all()
            for doc in docs:
                try:
                    tool_id = doc.get("id")
                    data = doc.get("profile", {})
                    profile = DatabaseToolProfile(id=tool_id, **data)
                    self.db_tools[tool_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load database tool {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.db_tools)} database tool profiles")
        except Exception as e:
            self.logger.error(f"Error loading database tools: {e}")

    def _save_db_tools(self) -> None:
        """Persist all database tool profiles to TinyDB."""
        try:
            self.db.truncate()
            for tool_id, profile in self.db_tools.items():
                self.db.insert(
                    {
                        "id": tool_id,
                        "profile": profile.model_dump(exclude={"id"}),
                    }
                )
            self.logger.info(f"Saved {len(self.db_tools)} database tool profiles")
        except Exception as e:
            self.logger.error(f"Error saving database tools: {e}")

    def _generate_id(self, name: str) -> str:
        """Generate a stable, URL-friendly id from the tool name."""
        base_id = name.strip().lower().replace(" ", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        if not base_id:
            base_id = "db_tool"

        candidate = base_id
        counter = 1
        while candidate in self.db_tools:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def _get_cache_key(self, tool_id: str) -> str:
        """Generate cache key for a tool."""
        return f"tool_{tool_id}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any], ttl_hours: float) -> bool:
        """Check if cache entry is still valid."""
        if not cache_entry:
            return False
        expires_at_str = cache_entry.get("expires_at")
        if not expires_at_str:
            return False
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            return datetime.now() < expires_at
        except Exception:
            return False

    def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        try:
            with self.cache_lock:
                current_time = datetime.now()
                expired_keys = []
                
                for key, entry in self.cache.items():
                    expires_at_str = entry.get("expires_at")
                    if expires_at_str:
                        try:
                            expires_at = datetime.fromisoformat(expires_at_str)
                            if current_time >= expires_at:
                                expired_keys.append(key)
                        except Exception:
                            expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                
                # Also clean up cache DB
                cache_docs = self.cache_db.all()
                for doc in cache_docs:
                    expires_at_str = doc.get("expires_at")
                    if expires_at_str:
                        try:
                            expires_at = datetime.fromisoformat(expires_at_str)
                            if current_time >= expires_at:
                                self.cache_db.remove(self.query.id == doc.get("id"))
                        except Exception:
                            pass
                
                if expired_keys:
                    self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        except Exception as e:
            self.logger.error(f"Error cleaning up expired cache: {e}")

    def _execute_query(self, profile: DatabaseToolProfile) -> Dict[str, Any]:
        """Execute database query and return results."""
        try:
            if profile.db_type == DatabaseType.MONGODB:
                return self._execute_mongodb_query(profile)
            else:
                return self._execute_sql_query(profile)
        except Exception as e:
            self.logger.error(f"Error executing query for {profile.id}: {e}")
            raise

    def _execute_sql_query(self, profile: DatabaseToolProfile) -> Dict[str, Any]:
        """Execute SQL query for SQL Server or MySQL."""
        try:
            config = profile.connection_config
            
            if profile.db_type == DatabaseType.MYSQL:
                try:
                    import pymysql
                except ImportError:
                    raise ValueError("MySQL driver not installed. Install with: pip install pymysql")
                
                # MySQL connection
                connection = pymysql.connect(
                    host=config.host,
                    port=config.port,
                    user=config.username,
                    password=config.password,
                    database=config.database,
                    cursorclass=pymysql.cursors.DictCursor,
                    **config.additional_params
                )
                
                try:
                    cursor = connection.cursor()
                    cursor.execute(profile.sql_statement)
                    
                    # Get column names
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    rows = cursor.fetchall()
                    # Convert dict rows to list of lists
                    data_rows = [[row[col] for col in columns] for row in rows] if rows else []
                    
                    return {
                        "columns": columns,
                        "rows": data_rows,
                        "total_rows": len(data_rows)
                    }
                finally:
                    connection.close()
                    
            elif profile.db_type == DatabaseType.SQLSERVER:
                try:
                    import pyodbc
                except ImportError:
                    raise ValueError("SQL Server driver not installed. Install with: pip install pyodbc")
                
                # SQL Server connection
                # Try different driver options
                drivers_to_try = [
                    "ODBC Driver 17 for SQL Server",
                    "ODBC Driver 18 for SQL Server",
                    "SQL Server",
                    "SQL Server Native Client 11.0"
                ]
                
                connection = None
                last_error = None
                
                for driver in drivers_to_try:
                    try:
                        conn_str = (
                            f"DRIVER={{{driver}}};"
                            f"SERVER={config.host},{config.port};"
                            f"DATABASE={config.database};"
                            f"UID={config.username};"
                            f"PWD={config.password}"
                        )
                        # Add additional params if any
                        for key, value in config.additional_params.items():
                            conn_str += f";{key}={value}"
                        
                        connection = pyodbc.connect(conn_str, timeout=30)
                        break
                    except Exception as e:
                        last_error = e
                        continue
                
                if not connection:
                    raise ValueError(f"Failed to connect to SQL Server. Tried drivers: {drivers_to_try}. Last error: {last_error}")
                
                try:
                    cursor = connection.cursor()
                    cursor.execute(profile.sql_statement)
                    
                    # Get column names
                    columns = [column[0] for column in cursor.description] if cursor.description else []
                    rows = cursor.fetchall()
                    data_rows = [list(row) for row in rows] if rows else []
                    
                    return {
                        "columns": columns,
                        "rows": data_rows,
                        "total_rows": len(data_rows)
                    }
                finally:
                    connection.close()
            else:
                raise ValueError(f"Unsupported SQL database type: {profile.db_type}")
                
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"SQL query execution error: {e}")
            raise

    def _execute_mongodb_query(self, profile: DatabaseToolProfile) -> Dict[str, Any]:
        """Execute MongoDB query."""
        try:
            from pymongo import MongoClient
            from bson import ObjectId
            
            config = profile.connection_config
            
            # Build connection string
            if config.username and config.password:
                connection_string = f"mongodb://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
            else:
                connection_string = f"mongodb://{config.host}:{config.port}/{config.database}"
            
            # Add additional params
            if config.additional_params:
                params_str = "&".join([f"{k}={v}" for k, v in config.additional_params.items()])
                connection_string += f"?{params_str}"
            
            client = MongoClient(connection_string)
            db = client[config.database]
            
            try:
                # Parse MongoDB query (assuming it's JSON)
                # Format: {"collection": "users", "query": {...}, "projection": {...}, "limit": 1000}
                try:
                    query_dict = json.loads(profile.sql_statement)
                except json.JSONDecodeError:
                    raise ValueError("MongoDB query must be valid JSON")
                
                if not isinstance(query_dict, dict):
                    raise ValueError("MongoDB query must be a JSON object")
                
                if "collection" not in query_dict:
                    raise ValueError("MongoDB query must include 'collection' field")
                
                collection_name = query_dict["collection"]
                query = query_dict.get("query", {})
                projection = query_dict.get("projection", None)
                limit = query_dict.get("limit", 1000)
                
                collection = db[collection_name]
                cursor = collection.find(query, projection).limit(limit)
                results = list(cursor)
                
                # Convert MongoDB documents to rows
                if results:
                    # Get all unique keys from all documents
                    all_keys = set()
                    for doc in results:
                        all_keys.update(doc.keys())
                    columns = sorted(list(all_keys))
                    
                    # Convert documents to rows, handling ObjectId and other BSON types
                    def convert_value(value):
                        if isinstance(value, ObjectId):
                            return str(value)
                        elif isinstance(value, dict):
                            return json.dumps(value, default=str)
                        elif isinstance(value, list):
                            return json.dumps(value, default=str)
                        return value
                    
                    rows = [[convert_value(doc.get(col)) for col in columns] for doc in results]
                else:
                    columns = []
                    rows = []
                
                return {
                    "columns": columns,
                    "rows": rows,
                    "total_rows": len(rows)
                }
            finally:
                client.close()
                
        except ImportError:
            raise ValueError("MongoDB driver not installed. Install with: pip install pymongo")
        except Exception as e:
            self.logger.error(f"MongoDB query execution error: {e}")
            raise

    def list_profiles(self) -> List[DatabaseToolProfile]:
        """List all database tool profiles."""
        return list(self.db_tools.values())

    def get_profile(self, tool_id: str) -> Optional[DatabaseToolProfile]:
        """Get a database tool profile by ID."""
        return self.db_tools.get(tool_id)

    def create_profile(self, req: DatabaseToolCreateRequest) -> str:
        """Create a new database tool profile."""
        tool_id = self._generate_id(req.name)
        profile = DatabaseToolProfile(
            id=tool_id,
            name=req.name,
            description=req.description,
            db_type=req.db_type,
            connection_config=req.connection_config,
            sql_statement=req.sql_statement,
            is_active=req.is_active,
            cache_ttl_hours=req.cache_ttl_hours,
            metadata=req.metadata or {},
        )
        self.db_tools[tool_id] = profile
        self._save_db_tools()
        self.logger.info(f"Created database tool profile: {tool_id}")
        return tool_id

    def update_profile(self, tool_id: str, req: DatabaseToolUpdateRequest) -> bool:
        """Update an existing database tool profile."""
        if tool_id not in self.db_tools:
            return False
        try:
            # Invalidate cache for this tool
            cache_key = self._get_cache_key(tool_id)
            with self.cache_lock:
                if cache_key in self.cache:
                    del self.cache[cache_key]
            
            # Get existing profile to preserve password if not provided
            existing_profile = self.db_tools[tool_id]
            existing_password = existing_profile.connection_config.password
            
            # Use provided password or keep existing
            update_config = req.connection_config
            if update_config.password is None or update_config.password == '':
                # Keep existing password
                connection_config = DatabaseConnectionConfig(
                    host=update_config.host,
                    port=update_config.port,
                    database=update_config.database,
                    username=update_config.username,
                    password=existing_password,
                    additional_params=update_config.additional_params,
                )
            else:
                # Use new password
                connection_config = DatabaseConnectionConfig(
                    host=update_config.host,
                    port=update_config.port,
                    database=update_config.database,
                    username=update_config.username,
                    password=update_config.password,
                    additional_params=update_config.additional_params,
                )
            
            profile = DatabaseToolProfile(
                id=tool_id,
                name=req.name,
                description=req.description,
                db_type=req.db_type,
                connection_config=connection_config,
                sql_statement=req.sql_statement,
                is_active=req.is_active,
                cache_ttl_hours=req.cache_ttl_hours,
                metadata=req.metadata or {},
            )
            self.db_tools[tool_id] = profile
            self._save_db_tools()
            self.logger.info(f"Updated database tool profile: {tool_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating database tool {tool_id}: {e}")
            return False

    def delete_profile(self, tool_id: str) -> bool:
        """Delete a database tool profile."""
        if tool_id not in self.db_tools:
            return False
        try:
            # Remove cache
            cache_key = self._get_cache_key(tool_id)
            with self.cache_lock:
                if cache_key in self.cache:
                    del self.cache[cache_key]
            
            del self.db_tools[tool_id]
            self.db.remove(self.query.id == tool_id)
            self.logger.info(f"Deleted database tool profile: {tool_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting database tool {tool_id}: {e}")
            return False

    def preview_data(self, tool_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get preview of first 10 rows of data from database query."""
        profile = self.db_tools.get(tool_id)
        if not profile:
            raise ValueError(f"Database tool {tool_id} not found")
        
        if not profile.is_active:
            raise ValueError(f"Database tool {tool_id} is not active")
        
        cache_key = self._get_cache_key(tool_id)
        cached = False
        
        # Check cache first
        if not force_refresh:
            with self.cache_lock:
                cache_entry = self.cache.get(cache_key)
                if cache_entry and self._is_cache_valid(cache_entry, profile.cache_ttl_hours):
                    cached = True
                    data = cache_entry.get("data", {})
                    expires_at = cache_entry.get("expires_at")
                    
                    # Return first 10 rows
                    rows = data.get("rows", [])[:10]
                    return {
                        "tool_id": tool_id,
                        "tool_name": profile.name,
                        "columns": data.get("columns", []),
                        "rows": rows,
                        "total_rows": data.get("total_rows"),
                        "cached": True,
                        "cache_expires_at": expires_at,
                        "metadata": {}
                    }
        
        # Execute query
        try:
            result = self._execute_query(profile)
            
            # Store in cache
            expires_at = datetime.now() + timedelta(hours=profile.cache_ttl_hours)
            cache_entry = {
                "data": result,
                "expires_at": expires_at.isoformat(),
                "created_at": datetime.now().isoformat()
            }
            
            with self.cache_lock:
                self.cache[cache_key] = cache_entry
                # Also save to cache DB
                self.cache_db.upsert(
                    {
                        "id": cache_key,
                        "tool_id": tool_id,
                        "data": result,
                        "expires_at": expires_at.isoformat(),
                        "created_at": datetime.now().isoformat()
                    },
                    self.query.id == cache_key
                )
            
            # Return first 10 rows
            rows = result.get("rows", [])[:10]
            return {
                "tool_id": tool_id,
                "tool_name": profile.name,
                "columns": result.get("columns", []),
                "rows": rows,
                "total_rows": result.get("total_rows"),
                "cached": False,
                "cache_expires_at": expires_at.isoformat(),
                "metadata": {}
            }
        except Exception as e:
            self.logger.error(f"Error previewing data for {tool_id}: {e}")
            raise

