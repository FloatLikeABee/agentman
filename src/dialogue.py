"""
Dialogue Service - Multi-turn conversation flows with system prompts
Similar to Customization but supports back-and-forth dialogue (max 5 turns)
"""
import logging
import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from tinydb import TinyDB, Query

from .config import settings
from .models import (
    DialogueProfile,
    DialogueCreateRequest,
    DialogueUpdateRequest,
    DialogueMessage,
    DialogueResponse,
    DialogueStartRequest,
    DialogueContinueRequest,
)


class DialogueManager:
    """Manage dialogue profiles and active conversations"""

    def __init__(self, rag_system=None, db_tools_manager=None, request_tools_manager=None):
        self.logger = logging.getLogger(__name__)
        self.dialogues: Dict[str, DialogueProfile] = {}
        self.active_conversations: Dict[str, Dict[str, Any]] = {}  # conversation_id -> conversation state
        self.rag_system = rag_system
        self.db_tools_manager = db_tools_manager
        self.request_tools_manager = request_tools_manager

        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "dialogues.json")
        self.db = TinyDB(self.db_path)
        self.query = Query()

        self._load_dialogues()

    def _load_dialogues(self) -> None:
        """Load dialogue profiles from TinyDB."""
        try:
            docs = self.db.all()
            for doc in docs:
                try:
                    dialogue_id = doc.get("id")
                    data = doc.get("profile", {})
                    # Ensure backward compatibility - add default empty lists for tool fields if missing
                    if "db_tools" not in data:
                        data["db_tools"] = []
                    if "request_tools" not in data:
                        data["request_tools"] = []
                    # Remove 'id' from data if present to avoid duplicate argument
                    if 'id' in data:
                        del data['id']
                    profile = DialogueProfile(id=dialogue_id, **data)
                    self.dialogues[dialogue_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load dialogue {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.dialogues)} dialogue profiles")
        except Exception as e:
            self.logger.error(f"Error loading dialogues: {e}")

    def _save_dialogues(self) -> None:
        """Persist all dialogue profiles to TinyDB."""
        try:
            self.db.truncate()
            for dialogue_id, profile in self.dialogues.items():
                self.db.insert(
                    {
                        "id": dialogue_id,
                        "profile": profile.model_dump(exclude={"id"}),
                    }
                )
            self.logger.info(f"Saved {len(self.dialogues)} dialogue profiles")
        except Exception as e:
            self.logger.error(f"Error saving dialogues: {e}")

    def _generate_id(self, name: str) -> str:
        """Generate a stable, URL-friendly id from the profile name."""
        base_id = name.strip().lower().replace(" ", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        if not base_id:
            base_id = "dialogue"

        candidate = base_id
        counter = 1
        while candidate in self.dialogues:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def list_profiles(self) -> List[DialogueProfile]:
        """List all dialogue profiles."""
        return list(self.dialogues.values())

    def get_profile(self, profile_id: str) -> Optional[DialogueProfile]:
        """Get a dialogue profile by ID."""
        return self.dialogues.get(profile_id)

    def create_profile(self, req: DialogueCreateRequest) -> str:
        """Create a new dialogue profile."""
        profile_id = self._generate_id(req.name)
        profile = DialogueProfile(
            id=profile_id,
            name=req.name,
            description=req.description,
            system_prompt=req.system_prompt,
            rag_collection=req.rag_collection,
            db_tools=req.db_tools or [],
            request_tools=req.request_tools or [],
            llm_provider=req.llm_provider,
            model_name=req.model_name,
            max_turns=req.max_turns,
            metadata=req.metadata or {},
        )
        self.dialogues[profile_id] = profile
        self._save_dialogues()
        self.logger.info(f"Created dialogue profile: {profile_id}")
        return profile_id

    def update_profile(self, profile_id: str, req: DialogueUpdateRequest) -> bool:
        """Update an existing dialogue profile."""
        if profile_id not in self.dialogues:
            return False
        try:
            profile = DialogueProfile(
                id=profile_id,
                name=req.name,
                description=req.description,
                system_prompt=req.system_prompt,
                rag_collection=req.rag_collection,
                db_tools=req.db_tools or [],
                request_tools=req.request_tools or [],
                llm_provider=req.llm_provider,
                model_name=req.model_name,
                max_turns=req.max_turns,
                metadata=req.metadata or {},
            )
            self.dialogues[profile_id] = profile
            self._save_dialogues()
            self.logger.info(f"Updated dialogue profile: {profile_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating dialogue {profile_id}: {e}")
            return False

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a dialogue profile."""
        if profile_id not in self.dialogues:
            return False
        try:
            del self.dialogues[profile_id]
            self.db.remove(self.query.id == profile_id)
            # Also clean up any active conversations for this profile
            conversations_to_remove = [
                conv_id for conv_id, conv in self.active_conversations.items()
                if conv.get("profile_id") == profile_id
            ]
            for conv_id in conversations_to_remove:
                del self.active_conversations[conv_id]
            self.logger.info(f"Deleted dialogue profile: {profile_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting dialogue {profile_id}: {e}")
            return False

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get an active conversation by ID."""
        return self.active_conversations.get(conversation_id)

    def _create_conversation(
        self,
        profile_id: str,
        initial_message: str,
        turn_number: int = 1
    ) -> str:
        """Create a new conversation session."""
        conversation_id = str(uuid.uuid4())
        profile = self.dialogues[profile_id]
        
        self.active_conversations[conversation_id] = {
            "conversation_id": conversation_id,
            "profile_id": profile_id,
            "turn_number": turn_number,
            "max_turns": profile.max_turns,
            "messages": [
                DialogueMessage(
                    role="user",
                    content=initial_message,
                    timestamp=datetime.now().isoformat()
                )
            ],
            "created_at": datetime.now().isoformat(),
        }
        return conversation_id

    def _add_message_to_conversation(
        self,
        conversation_id: str,
        role: str,
        content: str
    ) -> None:
        """Add a message to an existing conversation."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["messages"].append(
                DialogueMessage(
                    role=role,
                    content=content,
                    timestamp=datetime.now().isoformat()
                )
            )

    def _increment_turn(self, conversation_id: str) -> None:
        """Increment the turn number for a conversation."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["turn_number"] += 1

