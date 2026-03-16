import logging
import os
from typing import Dict

from tinydb import TinyDB, Query

from .config import settings
from .models import (
    SystemSettings,
    SystemPermissionsSettings,
    ExternalPlatformCredential,
    SystemSettingsResponse,
    SystemSettingsUpdateRequest,
)


class SystemSettingsManager:
    """Persist and manage high-level system settings separate from environment-based defaults.

    This manager stores overrides in TinyDB but does NOT expose raw secrets via the API.
    Tokens/passwords are stored and only their presence is reported back to the client.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "system_settings.json")
        self.db = TinyDB(self.db_path)
        self.query = Query()
        self._ensure_single_document()

    def _ensure_single_document(self) -> None:
        """Ensure we have a single settings document, seeding from config defaults if needed."""
        try:
            docs = self.db.all()
            if docs:
                return

            # Seed from existing config defaults
            providers_enabled: Dict[str, bool] = {
                "gemini": bool(settings.gemini_api_key),
                "qwen": bool(settings.qwen_api_key),
                "mistral": bool(settings.mistral_api_key),
                "groq": bool(settings.groq_api_key),
            }

            base_settings = SystemSettings(
                default_llm_provider=settings.default_llm_provider,
                default_model=settings.default_model,
                providers_enabled=providers_enabled,
                permissions=SystemPermissionsSettings(
                    allow_file_access=False,
                    allow_shell_commands=False,
                ),
                external_credentials={},
            )
            self.db.insert({"id": "system", "settings": base_settings.model_dump()})
        except Exception as e:
            self.logger.error(f"Error initializing system settings: {e}")

    def _load_settings(self) -> SystemSettings:
        """Load current settings from TinyDB."""
        try:
            doc = self.db.get(self.query.id == "system")
            if not doc:
                self._ensure_single_document()
                doc = self.db.get(self.query.id == "system")
            raw = doc.get("settings", {}) if doc else {}
            return SystemSettings(**raw)
        except Exception as e:
            self.logger.error(f"Error loading system settings, falling back to defaults: {e}")
            providers_enabled: Dict[str, bool] = {
                "gemini": bool(settings.gemini_api_key),
                "qwen": bool(settings.qwen_api_key),
                "mistral": bool(settings.mistral_api_key),
                "groq": bool(settings.groq_api_key),
            }
            return SystemSettings(
                default_llm_provider=settings.default_llm_provider,
                default_model=settings.default_model,
                providers_enabled=providers_enabled,
                permissions=SystemPermissionsSettings(
                    allow_file_access=False,
                    allow_shell_commands=False,
                ),
                external_credentials={},
            )

    def _save_settings(self, settings_obj: SystemSettings) -> None:
        """Persist settings back to TinyDB."""
        try:
            self.db.upsert(
                {"id": "system", "settings": settings_obj.model_dump()},
                self.query.id == "system",
            )
        except Exception as e:
            self.logger.error(f"Error saving system settings: {e}")

    def get_settings(self) -> SystemSettingsResponse:
        """Return current settings with token presence metadata."""
        settings_obj = self._load_settings()
        platform_has_token: Dict[str, bool] = {}
        for platform, cred in settings_obj.external_credentials.items():
            platform_has_token[platform] = bool(cred.access_token)

        # Mask tokens in the returned settings for safety
        masked_credentials: Dict[str, ExternalPlatformCredential] = {}
        for platform, cred in settings_obj.external_credentials.items():
            masked_credentials[platform] = ExternalPlatformCredential(
                platform=cred.platform,
                username=cred.username,
                access_token="***" if cred.access_token else None,
            )
        masked_settings = settings_obj.model_copy()
        masked_settings.external_credentials = masked_credentials

        return SystemSettingsResponse(
            settings=masked_settings,
            platform_has_token=platform_has_token,
        )

    def update_settings(self, update: SystemSettingsUpdateRequest) -> SystemSettingsResponse:
        """Apply a partial update to settings and return updated view."""
        current = self._load_settings()

        if update.default_llm_provider is not None:
            current.default_llm_provider = update.default_llm_provider
        if update.default_model is not None:
            current.default_model = update.default_model
        if update.providers_enabled is not None:
            # Merge flags
            merged = dict(current.providers_enabled)
            merged.update(update.providers_enabled)
            current.providers_enabled = merged
        if update.permissions is not None:
            current.permissions = update.permissions
        if update.external_credentials is not None:
            merged_creds: Dict[str, ExternalPlatformCredential] = dict(current.external_credentials)
            for platform, cred in update.external_credentials.items():
                # Empty string token means clear stored token
                token = cred.access_token
                if token == "":
                    token = None
                merged_creds[platform] = ExternalPlatformCredential(
                    platform=cred.platform or platform,
                    username=cred.username,
                    access_token=token,
                )
            current.external_credentials = merged_creds

        self._save_settings(current)
        return self.get_settings()

