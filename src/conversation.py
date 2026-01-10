"""
Conversation Service - Multi-AI conversation module
Allows two AI models to converse with each other and the user
"""
import logging
import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from tinydb import TinyDB, Query

from .config import settings
from .models import (
    ConversationConfig,
    ConversationCreateRequest,
    ConversationStartRequest,
    ConversationMessage,
    ConversationResponse,
    ConversationHistoryResponse,
    ConversationTurnRequest,
    LLMProviderType,
)
from .llm_factory import LLMFactory, LLMProvider
from .llm_langchain_wrapper import LangChainLLMWrapper
from .rag_system import RAGSystem
from .db_tools import DatabaseToolsManager
from .request_tools import RequestToolsManager


class ConversationProfile:
    """Stored conversation configuration profile"""
    def __init__(self, profile_id: str, name: str, description: Optional[str], config: ConversationConfig):
        self.id = profile_id
        self.name = name
        self.description = description
        self.config = config


class ConversationManager:
    """Manage conversation configurations and active sessions"""

    def __init__(self, rag_system: Optional[RAGSystem] = None, 
                 db_tools_manager: Optional[DatabaseToolsManager] = None,
                 request_tools_manager: Optional[RequestToolsManager] = None):
        self.logger = logging.getLogger(__name__)
        self.profiles: Dict[str, ConversationProfile] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session state
        self.rag_system = rag_system
        self.db_tools_manager = db_tools_manager
        self.request_tools_manager = request_tools_manager

        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "conversations.json")
        self.db = TinyDB(self.db_path)
        self.query = Query()
        
        # Create conversations folder for saved history
        self.conversations_dir = Path(settings.data_directory) / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)

        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load conversation profiles from TinyDB."""
        try:
            docs = self.db.all()
            for doc in docs:
                try:
                    profile_id = doc.get("id")
                    data = doc.get("profile", {})
                    config_data = data.get("config", {})
                    config = ConversationConfig.model_validate(config_data)
                    profile = ConversationProfile(
                        profile_id=profile_id,
                        name=data.get("name", ""),
                        description=data.get("description"),
                        config=config
                    )
                    self.profiles[profile_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load conversation profile {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.profiles)} conversation profiles")
        except Exception as e:
            self.logger.error(f"Error loading conversation profiles: {e}")

    def _save_profiles(self) -> None:
        """Persist all conversation profiles to TinyDB."""
        try:
            self.db.truncate()
            for profile_id, profile in self.profiles.items():
                self.db.insert({
                    "id": profile_id,
                    "profile": {
                        "name": profile.name,
                        "description": profile.description,
                        "config": profile.config.model_dump()
                    }
                })
            self.logger.info(f"Saved {len(self.profiles)} conversation profiles")
        except Exception as e:
            self.logger.error(f"Error saving conversation profiles: {e}")

    def _generate_id(self, name: str) -> str:
        """Generate a stable, URL-friendly id from the profile name."""
        base_id = name.strip().lower().replace(" ", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        if not base_id:
            base_id = "conversation"

        candidate = base_id
        counter = 1
        while candidate in self.profiles:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all conversation profiles."""
        return [
            {
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "config": profile.config.model_dump()
            }
            for profile in self.profiles.values()
        ]

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation profile by ID."""
        profile = self.profiles.get(profile_id)
        if not profile:
            return None
        return {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "config": profile.config.model_dump()
        }

    def create_profile(self, req: ConversationCreateRequest) -> str:
        """Create a new conversation profile."""
        profile_id = self._generate_id(req.name)
        profile = ConversationProfile(
            profile_id=profile_id,
            name=req.name,
            description=req.description,
            config=req.config
        )
        self.profiles[profile_id] = profile
        self._save_profiles()
        self.logger.info(f"Created conversation profile: {profile_id}")
        return profile_id

    def update_profile(self, profile_id: str, req: ConversationCreateRequest) -> bool:
        """Update an existing conversation profile."""
        if profile_id not in self.profiles:
            return False
        try:
            profile = ConversationProfile(
                profile_id=profile_id,
                name=req.name,
                description=req.description,
                config=req.config
            )
            self.profiles[profile_id] = profile
            self._save_profiles()
            self.logger.info(f"Updated conversation profile: {profile_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating conversation profile {profile_id}: {e}")
            return False

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a conversation profile."""
        if profile_id not in self.profiles:
            return False
        del self.profiles[profile_id]
        self._save_profiles()
        self.logger.info(f"Deleted conversation profile: {profile_id}")
        return True

    def start_conversation(self, req: ConversationStartRequest) -> ConversationResponse:
        """Start a new conversation session."""
        profile = self.profiles.get(req.config_id)
        if not profile:
            raise ValueError(f"Conversation profile not found: {req.config_id}")

        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Initialize conversation history with user topic
        initial_message = ConversationMessage(
            role="user",
            content=req.topic,
            timestamp=timestamp,
            turn_number=1
        )

        session_state = {
            "session_id": session_id,
            "config_id": req.config_id,
            "config_name": profile.name,
            "started_at": timestamp,
            "turn_number": 1,
            "max_turns": profile.config.max_turns,
            "conversation_history": [initial_message.model_dump()],
            "config": profile.config.model_dump()
        }

        self.active_sessions[session_id] = session_state

        # Start the conversation - first AI responds
        response = self._continue_conversation(session_id)
        return response

    def _continue_conversation(self, session_id: str, user_message: Optional[str] = None) -> ConversationResponse:
        """Continue a conversation turn."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        try:
            config = ConversationConfig.model_validate(session["config"])
        except Exception as e:
            self.logger.error(f"Error validating config: {e}, config data: {session.get('config')}")
            raise
        
        try:
            history = [ConversationMessage.model_validate(msg) for msg in session["conversation_history"]]
        except Exception as e:
            self.logger.error(f"Error validating history: {e}, history data: {session.get('conversation_history')}")
            raise
        turn_number = session["turn_number"]
        max_turns = session["max_turns"]

        # Add user message if provided
        if user_message:
            user_msg = ConversationMessage(
                role="user",
                content=user_message,
                timestamp=datetime.now().isoformat(),
                turn_number=turn_number
            )
            history.append(user_msg)
            session["conversation_history"].append(user_msg.model_dump())
            turn_number += 1
            session["turn_number"] = turn_number

        # Check if we've reached max turns
        if turn_number >= max_turns:
            session["ended_at"] = datetime.now().isoformat()
            self._save_conversation_history(session)
            return ConversationResponse(
                session_id=session_id,
                turn_number=turn_number,
                max_turns=max_turns,
                is_complete=True,
                messages=[],
                conversation_history=history,
                metadata={"message": "Conversation reached maximum turns"}
            )

        # Process ONE turn at a time for real-time updates
        # Models alternate: model1 -> model2 -> model1 -> model2...
        messages_this_turn = []
        request_logs = []  # Track all requests/responses for UI display
        
        # Process only one turn per call for real-time updates
        if turn_number < max_turns:
            # Determine which model should respond (alternate between model1 and model2)
            # After user message (turn 1), model1 responds (turn 2), then model2 (turn 3), etc.
            # turn 2, 4, 6... = model1 (even turns after user); turn 3, 5, 7... = model2 (odd turns after user)
            is_model1_turn = (turn_number % 2 == 0)  # turn 2, 4, 6... = model1; turn 3, 5, 7... = model2
            current_model_config = config.model1_config if is_model1_turn else config.model2_config
            role = "model1" if is_model1_turn else "model2"
            model_label = f"AI Model 1 ({config.model1_config.model_name})" if is_model1_turn else f"AI Model 2 ({config.model2_config.model_name})"

            # Get LLM caller for current model
            provider = LLMProvider.GEMINI if current_model_config.provider == LLMProviderType.GEMINI else (
                LLMProvider.QWEN if current_model_config.provider == LLMProviderType.QWEN else LLMProvider.MISTRAL
            )
            
            llm_caller = LLMFactory.create_caller(
                provider=provider,
                api_key=self._get_api_key(provider),
                model=current_model_config.model_name
            )

            # Get RAG context if configured for this model
            rag_context = ""
            if current_model_config.rag_collection and self.rag_system:
                try:
                    # Use the last message as query
                    query_text = history[-1].content if history else ""
                    rag_results = self.rag_system.query_collection(current_model_config.rag_collection, query_text, n_results=3)
                    if rag_results:
                        rag_context = "\n\nRelevant Context:\n" + "\n".join([r.get("content", "") for r in rag_results])
                except Exception as e:
                    self.logger.warning(f"Error fetching RAG context: {e}")

            # Build system prompt with context for this model
            system_prompt = current_model_config.system_prompt
            if rag_context:
                system_prompt += rag_context

            # Create LangChain wrapper
            llm = LangChainLLMWrapper(llm_caller=llm_caller)

            # Build conversation history string
            history_str = "\n".join([
                f"{'User' if msg.role == 'user' else (f'AI Model 1 ({config.model1_config.model_name})' if msg.role == 'model1' else f'AI Model 2 ({config.model2_config.model_name})')}: {msg.content}"
                for msg in history
            ])
            
            # Build the prompt
            full_prompt = f"{system_prompt}\n\nConversation History:\n{history_str}\n\n{model_label}:"

            # Log request to AI model
            self.logger.info(f"[Conversation] Request to {model_label} (Turn {turn_number}):")
            self.logger.info(f"[Conversation] Provider: {current_model_config.provider.value}, Model: {current_model_config.model_name}")
            if current_model_config.rag_collection:
                self.logger.info(f"[Conversation] RAG Collection: {current_model_config.rag_collection}")
            self.logger.info(f"[Conversation] System Prompt: {system_prompt[:200]}..." if len(system_prompt) > 200 else f"[Conversation] System Prompt: {system_prompt}")
            self.logger.info(f"[Conversation] Full Prompt Length: {len(full_prompt)} characters")

            # Generate response
            request_timestamp = datetime.now().isoformat()
            try:
                response_text = llm._call(full_prompt)
                response_timestamp = datetime.now().isoformat()
                self.logger.info(f"[Conversation] Response from {model_label} (Turn {turn_number}):")
                self.logger.info(f"[Conversation] Response Length: {len(response_text)} characters")
                self.logger.info(f"[Conversation] Response Preview: {response_text[:200]}..." if len(response_text) > 200 else f"[Conversation] Response: {response_text}")
                
                # Log request/response for UI
                request_logs.append({
                    "turn": turn_number,
                    "model": model_label,
                    "provider": current_model_config.provider.value,
                    "model_name": current_model_config.model_name,
                    "rag_collection": current_model_config.rag_collection,
                    "request_timestamp": request_timestamp,
                    "response_timestamp": response_timestamp,
                    "prompt_length": len(full_prompt),
                    "response_length": len(response_text),
                    "response_preview": response_text[:300] + "..." if len(response_text) > 300 else response_text,
                })
            except Exception as e:
                self.logger.error(f"Error generating response from {model_label}: {e}")
                request_logs.append({
                    "turn": turn_number,
                    "model": model_label,
                    "provider": current_model_config.provider.value,
                    "model_name": current_model_config.model_name,
                    "error": str(e),
                    "request_timestamp": request_timestamp,
                })
                raise
            
            # Create response message
            response_msg = ConversationMessage(
                role=role,
                content=response_text,
                timestamp=datetime.now().isoformat(),
                turn_number=turn_number
            )
            history.append(response_msg)
            messages_this_turn.append(response_msg)
            session["conversation_history"].append(response_msg.model_dump())
            turn_number += 1
            session["turn_number"] = turn_number

        # Check if we've reached max turns
        is_complete = turn_number >= max_turns
        if is_complete:
            session["ended_at"] = datetime.now().isoformat()
            self._save_conversation_history(session)

        # Return response with all messages from this turn
        return ConversationResponse(
            session_id=session_id,
            turn_number=turn_number,
            max_turns=max_turns,
            is_complete=is_complete,
            messages=messages_this_turn,
            conversation_history=history,
            metadata={
                "total_messages_this_turn": len(messages_this_turn),
                "request_logs": request_logs
            }
        )

    def continue_conversation(self, req: ConversationTurnRequest) -> ConversationResponse:
        """Continue an existing conversation."""
        return self._continue_conversation(req.session_id, req.user_message)

    def _get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Get API key for the provider."""
        from .config import settings
        if provider == LLMProvider.GEMINI:
            return settings.gemini_api_key
        elif provider == LLMProvider.QWEN:
            return settings.qwen_api_key
        elif provider == LLMProvider.MISTRAL:
            return settings.mistral_api_key
        return None

    def _save_conversation_history(self, session: Dict[str, Any]) -> str:
        """Save conversation history to a text file."""
        try:
            session_id = session["session_id"]
            config_name = session["config_name"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create safe filename
            safe_name = "".join(c for c in config_name if c.isalnum() or c in ("_", "-", " "))
            safe_name = safe_name.replace(" ", "_")
            filename = f"{safe_name}_{session_id[:8]}_{timestamp}.txt"
            filepath = self.conversations_dir / filename

            # Format conversation history
            lines = []
            lines.append("=" * 80)
            lines.append(f"CONVERSATION HISTORY")
            lines.append("=" * 80)
            lines.append(f"Session ID: {session_id}")
            lines.append(f"Configuration: {session['config_name']}")
            lines.append(f"Started: {session['started_at']}")
            if session.get("ended_at"):
                lines.append(f"Ended: {session['ended_at']}")
            lines.append(f"Total Turns: {session['turn_number']}")
            lines.append("=" * 80)
            lines.append("")

            for msg_data in session["conversation_history"]:
                msg = ConversationMessage.model_validate(msg_data)
                config_data = session['config']
                model1_name = config_data.get('model1_config', {}).get('model_name', 'Model 1')
                model2_name = config_data.get('model2_config', {}).get('model_name', 'Model 2')
                role_display = {
                    "user": "USER",
                    "model1": f"AI MODEL 1 ({model1_name})",
                    "model2": f"AI MODEL 2 ({model2_name})"
                }.get(msg.role, msg.role.upper())
                
                lines.append(f"[Turn {msg.turn_number}] {role_display} ({msg.timestamp})")
                lines.append("-" * 80)
                lines.append(msg.content)
                lines.append("")

            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            self.logger.info(f"Saved conversation history to {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Error saving conversation history: {e}")
            raise

    def get_conversation_history(self, session_id: str) -> Optional[ConversationHistoryResponse]:
        """Get conversation history for a session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        saved_file_path = None
        if session.get("ended_at"):
            # Try to find saved file
            try:
                for filepath in self.conversations_dir.glob(f"*{session_id[:8]}*.txt"):
                    saved_file_path = str(filepath)
                    break
            except Exception:
                pass

        history = [ConversationMessage.model_validate(msg) for msg in session["conversation_history"]]
        
        return ConversationHistoryResponse(
            session_id=session_id,
            config_id=session["config_id"],
            config_name=session["config_name"],
            started_at=session["started_at"],
            ended_at=session.get("ended_at"),
            total_turns=session["turn_number"],
            conversation_history=history,
            saved_file_path=saved_file_path
        )

    def list_saved_conversations(self) -> List[Dict[str, Any]]:
        """List all saved conversation files."""
        conversations = []
        try:
            for filepath in sorted(self.conversations_dir.glob("*.txt"), reverse=True):
                stat = filepath.stat()
                conversations.append({
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        except Exception as e:
            self.logger.error(f"Error listing saved conversations: {e}")
        return conversations

    def get_saved_conversation_content(self, filename: str) -> Optional[str]:
        """Get the content of a saved conversation file."""
        try:
            filepath = self.conversations_dir / filename
            if not filepath.exists() or not filepath.is_file():
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading saved conversation file {filename}: {e}")
            return None
