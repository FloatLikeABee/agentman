"""
Crawler Manager - Manage crawler profiles
"""
import logging
import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from tinydb import TinyDB, Query

from .config import settings
from .models import (
    CrawlerProfile,
    CrawlerCreateRequest,
    CrawlerUpdateRequest,
)


class CrawlerManager:
    """Manage crawler profiles"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.profiles: Dict[str, CrawlerProfile] = {}

        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "crawler_profiles.json")
        self.db = TinyDB(self.db_path)
        self.query = Query()

        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load crawler profiles from TinyDB."""
        try:
            docs = self.db.all()
            for doc in docs:
                try:
                    profile_id = doc.get("id")
                    data = doc.get("profile", {})
                    profile = CrawlerProfile(id=profile_id, **data)
                    self.profiles[profile_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load crawler profile {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.profiles)} crawler profiles")
        except Exception as e:
            self.logger.error(f"Error loading crawler profiles: {e}")

    def _save_profiles(self) -> None:
        """Persist all crawler profiles to TinyDB."""
        try:
            self.db.truncate()
            for profile_id, profile in self.profiles.items():
                self.db.insert({
                    "id": profile_id,
                    "profile": profile.model_dump()
                })
            self.logger.info(f"Saved {len(self.profiles)} crawler profiles")
        except Exception as e:
            self.logger.error(f"Error saving crawler profiles: {e}")

    def _generate_id(self, name: str) -> str:
        """Generate a stable, URL-friendly id from the profile name."""
        base_id = name.strip().lower().replace(" ", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        if not base_id:
            base_id = "crawler"

        candidate = base_id
        counter = 1
        while candidate in self.profiles:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all crawler profiles."""
        return [
            {
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "url": profile.url,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
            }
            for profile in self.profiles.values()
        ]

    def get_profile(self, profile_id: str) -> Optional[CrawlerProfile]:
        """Get a crawler profile by ID."""
        return self.profiles.get(profile_id)

    def create_profile(self, req: CrawlerCreateRequest) -> CrawlerProfile:
        """Create a new crawler profile."""
        profile_id = self._generate_id(req.name)
        timestamp = datetime.now().isoformat()
        
        profile = CrawlerProfile(
            id=profile_id,
            name=req.name,
            description=req.description,
            url=req.url,
            use_js=req.use_js,
            llm_provider=req.llm_provider,
            model=req.model,
            collection_name=req.collection_name,
            collection_description=req.collection_description,
            follow_links=req.follow_links,
            max_depth=req.max_depth,
            max_pages=req.max_pages,
            same_domain_only=req.same_domain_only,
            headers=req.headers,
            created_at=timestamp,
            updated_at=timestamp,
        )
        
        self.profiles[profile_id] = profile
        self._save_profiles()
        self.logger.info(f"Created crawler profile: {profile_id}")
        return profile

    def update_profile(self, profile_id: str, req: CrawlerUpdateRequest) -> Optional[CrawlerProfile]:
        """Update an existing crawler profile."""
        profile = self.profiles.get(profile_id)
        if not profile:
            return None

        # Update only provided fields
        update_data = req.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            if value is not None:
                setattr(profile, key, value)
        
        profile.updated_at = datetime.now().isoformat()
        self._save_profiles()
        self.logger.info(f"Updated crawler profile: {profile_id}")
        return profile

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a crawler profile."""
        if profile_id not in self.profiles:
            return False
        
        del self.profiles[profile_id]
        self.db.remove(self.query.id == profile_id)
        self.logger.info(f"Deleted crawler profile: {profile_id}")
        return True
