import logging
import os
from typing import List, Optional, Dict, Any

from tinydb import TinyDB, Query

from .config import settings
from .models import (
    CustomizationProfile,
    CustomizationCreateRequest,
)


class CustomizationManager:
    """Manage customization profiles (instruction + optional RAG context)."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.customizations: Dict[str, CustomizationProfile] = {}

        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "customizations.json")
        self.db = TinyDB(self.db_path)
        self.query = Query()

        self._load_customizations()

    def _load_customizations(self) -> None:
        """Load customization profiles from TinyDB."""
        try:
            docs = self.db.all()
            for doc in docs:
                try:
                    profile_id = doc.get("id")
                    data = doc.get("profile", {})
                    profile = CustomizationProfile(id=profile_id, **data)
                    self.customizations[profile_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load customization {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.customizations)} customization profiles")
        except Exception as e:
            self.logger.error(f"Error loading customizations: {e}")

    def _save_customizations(self) -> None:
        """Persist all customization profiles to TinyDB."""
        try:
            self.db.truncate()
            for profile_id, profile in self.customizations.items():
                self.db.insert(
                    {
                        "id": profile_id,
                        "profile": profile.model_dump(exclude={"id"}),
                    }
                )
            self.logger.info(f"Saved {len(self.customizations)} customization profiles")
        except Exception as e:
            self.logger.error(f"Error saving customizations: {e}")

    def _generate_id(self, name: str) -> str:
        """Generate a stable, URL-friendly id from the profile name."""
        base_id = name.strip().lower().replace(" ", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        if not base_id:
            base_id = "customization"

        candidate = base_id
        counter = 1
        while candidate in self.customizations:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def list_profiles(self) -> List[CustomizationProfile]:
        return list(self.customizations.values())

    def get_profile(self, profile_id: str) -> Optional[CustomizationProfile]:
        return self.customizations.get(profile_id)

    def create_profile(self, req: CustomizationCreateRequest) -> str:
        profile_id = self._generate_id(req.name)
        profile = CustomizationProfile(
            id=profile_id,
            name=req.name,
            description=req.description,
            system_prompt=req.system_prompt,
            rag_collection=req.rag_collection,
            llm_provider=req.llm_provider,
            model_name=req.model_name,
            metadata=req.metadata or {},
        )
        self.customizations[profile_id] = profile
        self._save_customizations()
        self.logger.info(f"Created customization profile: {profile_id}")
        return profile_id

    def update_profile(self, profile_id: str, req: CustomizationCreateRequest) -> bool:
        """Update an existing customization profile."""
        if profile_id not in self.customizations:
            return False
        try:
            # Update the profile with new data
            profile = CustomizationProfile(
                id=profile_id,  # Keep the same ID
                name=req.name,
                description=req.description,
                system_prompt=req.system_prompt,
                rag_collection=req.rag_collection,
                llm_provider=req.llm_provider,
                model_name=req.model_name,
                metadata=req.metadata or {},
            )
            self.customizations[profile_id] = profile
            self._save_customizations()
            self.logger.info(f"Updated customization profile: {profile_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating customization {profile_id}: {e}")
            return False

    def delete_profile(self, profile_id: str) -> bool:
        if profile_id not in self.customizations:
            return False
        try:
            del self.customizations[profile_id]
            self.db.remove(self.query.id == profile_id)
            self.logger.info(f"Deleted customization profile: {profile_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting customization {profile_id}: {e}")
            return False


