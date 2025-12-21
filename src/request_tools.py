import logging
import os
import json
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
import threading

import requests
from tinydb import TinyDB, Query

from .config import settings
from .models import (
    RequestProfile,
    RequestCreateRequest,
    RequestUpdateRequest,
    RequestType,
    HTTPMethod,
)


class RequestToolsManager:
    """Manage request configurations and execute HTTP/internal service calls."""

    def __init__(self, api_instance=None):
        self.logger = logging.getLogger(__name__)
        self.requests: Dict[str, RequestProfile] = {}
        self.lock = threading.Lock()
        self.api_instance = api_instance  # Reference to API instance for internal calls

        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "request_tools.json")
        self.db = TinyDB(self.db_path)
        self.query = Query()

        self._load_requests()

    def _load_requests(self) -> None:
        """Load request profiles from TinyDB."""
        try:
            docs = self.db.all()
            for doc in docs:
                try:
                    request_id = doc.get("id")
                    data = doc.get("profile", {})
                    profile = RequestProfile(id=request_id, **data)
                    self.requests[request_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load request {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.requests)} request profiles")
        except Exception as e:
            self.logger.error(f"Error loading requests: {e}")

    def _save_requests(self) -> None:
        """Persist all request profiles to TinyDB."""
        try:
            with self.lock:
                self.db.truncate()
                for request_id, profile in self.requests.items():
                    self.db.insert(
                        {
                            "id": request_id,
                            "profile": profile.model_dump(exclude={"id"}),
                        }
                    )
                self.logger.info(f"Saved {len(self.requests)} request profiles")
        except Exception as e:
            self.logger.error(f"Error saving requests: {e}")

    def _generate_id(self, name: str) -> str:
        """Generate a stable, URL-friendly id from the request name."""
        base_id = name.strip().lower().replace(" ", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        if not base_id:
            base_id = "request"

        candidate = base_id
        counter = 1
        while candidate in self.requests:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def _check_name_unique(self, name: str, exclude_id: Optional[str] = None) -> bool:
        """Check if request name is unique."""
        for req_id, profile in self.requests.items():
            if exclude_id and req_id == exclude_id:
                continue
            if profile.name.lower() == name.lower():
                return False
        return True

    def list_profiles(self) -> List[RequestProfile]:
        """List all request profiles."""
        return list(self.requests.values())

    def get_profile(self, request_id: str) -> Optional[RequestProfile]:
        """Get a request profile by ID."""
        return self.requests.get(request_id)

    def get_profile_by_name(self, name: str) -> Optional[RequestProfile]:
        """Get a request profile by name."""
        for profile in self.requests.values():
            if profile.name.lower() == name.lower():
                return profile
        return None

    def create_profile(self, req: RequestCreateRequest) -> str:
        """Create a new request profile."""
        # Check name uniqueness
        if not self._check_name_unique(req.name):
            raise ValueError(f"Request name '{req.name}' already exists. Names must be unique.")

        # Validate request configuration
        if req.request_type == RequestType.HTTP:
            if not req.method:
                raise ValueError("HTTP method is required for HTTP requests")
            if not req.url:
                raise ValueError("URL is required for HTTP requests")
        elif req.request_type == RequestType.INTERNAL:
            if not req.endpoint:
                raise ValueError("Endpoint is required for internal requests")

        request_id = self._generate_id(req.name)
        profile = RequestProfile(
            id=request_id,
            name=req.name,
            description=req.description,
            request_type=req.request_type,
            method=req.method,
            url=req.url,
            endpoint=req.endpoint,
            headers=req.headers,
            params=req.params,
            body=req.body,
            timeout=req.timeout,
            metadata=req.metadata,
        )
        self.requests[request_id] = profile
        self._save_requests()
        self.logger.info(f"Created request profile: {request_id}")
        return request_id

    def update_profile(self, request_id: str, req: RequestUpdateRequest) -> bool:
        """Update an existing request profile."""
        if request_id not in self.requests:
            return False

        # Check name uniqueness (excluding current request)
        if not self._check_name_unique(req.name, exclude_id=request_id):
            raise ValueError(f"Request name '{req.name}' already exists. Names must be unique.")

        # Validate request configuration
        if req.request_type == RequestType.HTTP:
            if not req.method:
                raise ValueError("HTTP method is required for HTTP requests")
            if not req.url:
                raise ValueError("URL is required for HTTP requests")
        elif req.request_type == RequestType.INTERNAL:
            if not req.endpoint:
                raise ValueError("Endpoint is required for internal requests")

        try:
            # Preserve last response
            existing_profile = self.requests[request_id]
            last_response = existing_profile.last_response
            last_executed_at = existing_profile.last_executed_at

            profile = RequestProfile(
                id=request_id,
                name=req.name,
                description=req.description,
                request_type=req.request_type,
                method=req.method,
                url=req.url,
                endpoint=req.endpoint,
                headers=req.headers,
                params=req.params,
                body=req.body,
                timeout=req.timeout,
                last_response=last_response,  # Preserve last response
                last_executed_at=last_executed_at,  # Preserve last execution time
                metadata=req.metadata,
            )
            self.requests[request_id] = profile
            self._save_requests()
            self.logger.info(f"Updated request profile: {request_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating request {request_id}: {e}")
            raise

    def delete_profile(self, request_id: str) -> bool:
        """Delete a request profile."""
        if request_id not in self.requests:
            return False
        try:
            del self.requests[request_id]
            self.db.remove(self.query.id == request_id)
            self.logger.info(f"Deleted request profile: {request_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting request {request_id}: {e}")
            return False

    def _execute_http_request(self, profile: RequestProfile) -> Dict[str, Any]:
        """Execute an HTTP request."""
        start_time = time.time()
        try:
            # Prepare request parameters
            method = profile.method.value if profile.method else "GET"
            url = profile.url
            headers = profile.headers or {}
            params = profile.params or {}
            timeout = profile.timeout

            # Prepare body
            data = None
            json_data = None
            if profile.body:
                if isinstance(profile.body, dict) or isinstance(profile.body, list):
                    json_data = profile.body
                elif isinstance(profile.body, str):
                    try:
                        # Try to parse as JSON
                        json_data = json.loads(profile.body)
                    except json.JSONDecodeError:
                        # If not JSON, send as plain text
                        data = profile.body
                        headers.setdefault("Content-Type", "text/plain")

            # Make the request
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                data=data,
                timeout=timeout,
            )

            execution_time = time.time() - start_time

            # Parse response
            try:
                response_data = response.json()
            except ValueError:
                response_data = response.text

            return {
                "success": True,
                "status_code": response.status_code,
                "response_data": response_data,
                "response_headers": dict(response.headers),
                "execution_time": execution_time,
                "error": None,
            }
        except requests.exceptions.Timeout:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "status_code": None,
                "response_data": None,
                "response_headers": {},
                "execution_time": execution_time,
                "error": f"Request timed out after {profile.timeout} seconds",
            }
        except requests.exceptions.RequestException as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "status_code": None,
                "response_data": None,
                "response_headers": {},
                "execution_time": execution_time,
                "error": str(e),
            }
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error executing HTTP request {profile.id}: {e}")
            return {
                "success": False,
                "status_code": None,
                "response_data": None,
                "response_headers": {},
                "execution_time": execution_time,
                "error": str(e),
            }

    def _execute_internal_request(self, profile: RequestProfile) -> Dict[str, Any]:
        """Execute an internal service call."""
        start_time = time.time()
        try:
            if not self.api_instance:
                raise ValueError("API instance not available for internal requests")

            endpoint = profile.endpoint
            if not endpoint:
                raise ValueError("Endpoint is required for internal requests")

            # Remove leading slash if present
            endpoint = endpoint.lstrip("/")

            # Prepare request data
            body = profile.body
            params = profile.params or {}

            # Call internal endpoint
            # This is a simplified approach - in practice, you might want to route through FastAPI's app
            # For now, we'll use a direct call approach
            result = {
                "success": False,
                "status_code": None,
                "response_data": None,
                "response_headers": {},
                "execution_time": 0,
                "error": "Internal service calls require API instance routing",
            }

            # Note: Internal service calls would need to be routed through the FastAPI app
            # This is a placeholder - actual implementation would depend on your routing needs
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["error"] = "Internal service call routing not yet implemented. Use HTTP requests for external APIs."

            return result
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error executing internal request {profile.id}: {e}")
            return {
                "success": False,
                "status_code": None,
                "response_data": None,
                "response_headers": {},
                "execution_time": execution_time,
                "error": str(e),
            }

    def execute_request(self, request_id: str) -> Dict[str, Any]:
        """Execute a request and save the response."""
        profile = self.requests.get(request_id)
        if not profile:
            raise ValueError(f"Request {request_id} not found")

        # Execute based on request type
        if profile.request_type == RequestType.HTTP:
            result = self._execute_http_request(profile)
        elif profile.request_type == RequestType.INTERNAL:
            result = self._execute_internal_request(profile)
        else:
            raise ValueError(f"Unknown request type: {profile.request_type}")

        # Save response to profile
        executed_at = datetime.now().isoformat()
        profile.last_response = result
        profile.last_executed_at = executed_at

        # Save to database
        self._save_requests()

        # Return execution response
        return {
            "request_id": request_id,
            "request_name": profile.name,
            "success": result["success"],
            "status_code": result.get("status_code"),
            "response_data": result.get("response_data"),
            "response_headers": result.get("response_headers", {}),
            "execution_time": result["execution_time"],
            "error": result.get("error"),
            "executed_at": executed_at,
            "metadata": {},
        }

