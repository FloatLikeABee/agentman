import json
import logging
import os
from typing import Dict, List, Optional, Any

from tinydb import TinyDB, Query

from .config import settings
from .models import (
    AdviserCreateRequest,
    AdviserProfile,
    AdviserFileInput,
    LLMProviderType,
    RAGDataInput,
    DataFormat,
    AgentConfig,
    AgentType,
)
from .rag_system import RAGSystem
from .tools import ToolManager
from .llm_factory import LLMFactory, LLMProvider
from .agent_manager import AgentManager


class AdviserManager:
    """Manage Advisers: higher-level helpers built on top of Agents + RAG + Web Search."""

    def __init__(
        self,
        rag_system: RAGSystem,
        tool_manager: ToolManager,
        agent_manager: AgentManager,
    ):
        self.logger = logging.getLogger(__name__)
        self.rag_system = rag_system
        self.tool_manager = tool_manager
        self.agent_manager = agent_manager

        self.advisers: Dict[str, AdviserProfile] = {}

        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "advisers.json")
        self.db = TinyDB(self.db_path)
        self.query = Query()

        self._load_advisers()

    def _load_advisers(self) -> None:
        """Load adviser profiles from TinyDB."""
        try:
            docs = self.db.all()
            for doc in docs:
                try:
                    adviser_id = doc.get("id")
                    data = doc.get("profile", {})
                    if "id" in data:
                        del data["id"]
                    profile = AdviserProfile(id=adviser_id, **data)
                    self.advisers[adviser_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load adviser {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.advisers)} advisers")
        except Exception as e:
            self.logger.error(f"Error loading advisers: {e}")

    def _save_advisers(self) -> None:
        """Persist all adviser profiles to TinyDB."""
        try:
            self.db.truncate()
            for adviser_id, profile in self.advisers.items():
                self.db.insert(
                    {
                        "id": adviser_id,
                        "profile": profile.model_dump(exclude={"id"}),
                    }
                )
            self.logger.info(f"Saved {len(self.advisers)} advisers")
        except Exception as e:
            self.logger.error(f"Error saving advisers: {e}")

    def _generate_id(self, name: str) -> str:
        """Generate a stable, URL-friendly id from the adviser name."""
        base_id = name.strip().lower().replace(" ", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        if not base_id:
            base_id = "adviser"

        candidate = base_id
        counter = 1
        while candidate in self.advisers:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def list_advisers(self) -> List[AdviserProfile]:
        return list(self.advisers.values())

    def get_adviser(self, adviser_id: str) -> Optional[AdviserProfile]:
        return self.advisers.get(adviser_id)

    def delete_adviser(self, adviser_id: str) -> bool:
        """Delete an adviser and its underlying agent (RAG collections are preserved)."""
        if adviser_id not in self.advisers:
            return False
        try:
            profile = self.advisers[adviser_id]
            agent_id = profile.agent_id
            if agent_id:
                try:
                    self.agent_manager.delete_agent(agent_id)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to delete underlying agent '{agent_id}' for adviser '{adviser_id}': {e}"
                    )
            del self.advisers[adviser_id]
            self.db.remove(self.query.id == adviser_id)
            self.logger.info(f"Deleted adviser: {adviser_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting adviser {adviser_id}: {e}")
            return False

    def _default_provider_and_model(
        self, provider_override: Optional[LLMProviderType], model_override: Optional[str]
    ) -> (LLMProviderType, str, str):
        """Resolve provider enum, API key, and model name with sensible defaults."""
        provider_str = (
            provider_override.value if provider_override else settings.default_llm_provider
        )

        if provider_str == "gemini":
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = model_override or settings.gemini_default_model
        elif provider_str == "qwen":
            provider = LLMProviderType.QWEN
            api_key = settings.qwen_api_key
            model_name = model_override or settings.qwen_default_model
        elif provider_str == "mistral":
            provider = LLMProviderType.MISTRAL
            api_key = settings.mistral_api_key
            model_name = model_override or settings.mistral_default_model
        elif provider_str == "groq":
            provider = LLMProviderType.GROQ
            api_key = getattr(settings, "groq_api_key", "")
            model_name = model_override or getattr(
                settings, "groq_default_model", "llama-3.3-70b-versatile"
            )
        else:
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = model_override or settings.gemini_default_model

        return provider, api_key, model_name

    def _normalize_prompt_and_description(
        self,
        draft_prompt: str,
        draft_description: Optional[str],
        provider_override: Optional[LLMProviderType],
        model_override: Optional[str],
    ) -> Dict[str, str]:
        """Use LLM to clean up system prompt and ensure we have a solid description."""
        provider, api_key, model_name = self._default_provider_and_model(
            provider_override, model_override
        )

        try:
            llm_caller = LLMFactory.create_caller(
                provider=LLMProvider(provider.value),
                api_key=api_key,
                model=model_name,
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize LLM for adviser normalization, using raw prompt/description. Error: {e}"
            )
            return {
                "system_prompt": draft_prompt,
                "description": draft_description
                or self._fallback_description_from_prompt(draft_prompt),
            }

        payload = {
            "prompt": draft_prompt,
            "description": draft_description or "",
        }
        normalization_instructions = (
            "You are an AI assistant that rewrites agent system prompts and descriptions "
            "to be clear, professional, safe, and aligned with best practices.\n\n"
            "Given the following JSON with a draft system prompt and an optional description, "
            "you MUST respond with a single, valid JSON object ONLY, with this exact shape:\n"
            '{\n  "system_prompt": "cleaned and improved system prompt text",\n'
            '  "description": "short, user-facing description of what this adviser does"\n}\n\n'
            "- Do not add explanations.\n"
            "- Do not add extra fields.\n"
            "- Keep the description concise (1–2 sentences).\n"
        )

        full_prompt = (
            f"{normalization_instructions}\n\n"
            f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            raw = llm_caller.generate(full_prompt)
            # Strip common Markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                # In case model added a language tag like ```json
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1]
            data = json.loads(cleaned)
            system_prompt = (
                str(data.get("system_prompt")).strip() if data.get("system_prompt") else draft_prompt
            )
            description = (
                str(data.get("description")).strip()
                if data.get("description")
                else draft_description or self._fallback_description_from_prompt(draft_prompt)
            )
            return {"system_prompt": system_prompt, "description": description}
        except Exception as e:
            self.logger.warning(
                f"Failed to parse normalization response, using raw prompt/description. Error: {e}"
            )
            return {
                "system_prompt": draft_prompt,
                "description": draft_description
                or self._fallback_description_from_prompt(draft_prompt),
            }

    def _fallback_description_from_prompt(self, prompt: str) -> str:
        snippet = prompt.strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177].rstrip() + "..."
        return f"Adviser configured with system instructions: {snippet}"

    def _ingest_files_to_collection(
        self, adviser_id: str, adviser_name: str, files: List[AdviserFileInput]
    ) -> Optional[str]:
        """Create (or reuse) an adviser-specific RAG collection and ingest uploaded files."""
        if not files:
            return None

        collection_name = f"adviser_{adviser_id}_kb"
        for file in files:
            try:
                data_format = file.format
                if isinstance(data_format, str):
                    # Normalize from raw string to enum if needed
                    data_format = DataFormat(data_format)

                rag_input = RAGDataInput(
                    name=file.filename,
                    description=file.description
                    or f"Base knowledge file for adviser '{adviser_name}'",
                    format=data_format,
                    content=file.content,
                    tags=[adviser_id, "adviser"],
                    metadata={"adviser_id": adviser_id, "filename": file.filename},
                )
                success = self.rag_system.add_data_to_collection(collection_name, rag_input)
                if not success:
                    self.logger.warning(
                        f"Failed to add data from file '{file.filename}' to collection '{collection_name}'"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error ingesting file '{file.filename}' for adviser '{adviser_id}': {e}"
                )

        return collection_name

    def _build_agent_config(
        self,
        name: str,
        description: str,
        system_prompt: str,
        provider: LLMProviderType,
        model_name: str,
        rag_collections: List[str],
    ) -> AgentConfig:
        """Create an AgentConfig for this adviser, always enabling web search and RAG."""
        tools: List[str] = ["web_search"]

        return AgentConfig(
            name=name,
            description=description,
            agent_type=AgentType.HYBRID,
            llm_provider=provider,
            model_name=model_name,
            temperature=0.7,
            max_tokens=8192,
            rag_collections=rag_collections,
            tools=tools,
            system_prompt=system_prompt,
            system_prompt_data=None,
            is_active=True,
        )

    def create_adviser(self, req: AdviserCreateRequest) -> str:
        """Create a new adviser, including its base RAG data and underlying agent."""
        if not req.draft_system_prompt or not req.draft_system_prompt.strip():
            raise ValueError("draft_system_prompt is required to create an adviser")

        adviser_id = self._generate_id(req.name)

        # Normalize prompt and description via LLM
        normalized = self._normalize_prompt_and_description(
            draft_prompt=req.draft_system_prompt,
            draft_description=req.description,
            provider_override=req.llm_provider,
            model_override=req.model_name,
        )
        system_prompt = normalized["system_prompt"]
        final_description = normalized["description"]

        # Resolve provider/model
        provider, _, resolved_model_name = self._default_provider_and_model(
            req.llm_provider, req.model_name
        )

        # Ingest uploaded files into adviser-specific collection
        base_collection = self._ingest_files_to_collection(
            adviser_id=adviser_id,
            adviser_name=req.name,
            files=req.files,
        )

        rag_collections: List[str] = list(req.existing_rag_collections or [])
        if base_collection:
            rag_collections.append(base_collection)
        # Deduplicate while preserving order
        seen: set = set()
        deduped_rag_collections: List[str] = []
        for col in rag_collections:
            if col and col not in seen:
                seen.add(col)
                deduped_rag_collections.append(col)

        # Create underlying agent with RAG + Web Search
        agent_config = self._build_agent_config(
            name=req.name,
            description=final_description,
            system_prompt=system_prompt,
            provider=provider,
            model_name=resolved_model_name,
            rag_collections=deduped_rag_collections,
        )
        agent_id = self.agent_manager.create_agent(agent_config)

        profile = AdviserProfile(
            id=adviser_id,
            name=req.name,
            description=final_description,
            system_prompt=system_prompt,
            rag_collections=deduped_rag_collections,
            base_collection=base_collection,
            llm_provider=provider,
            model_name=resolved_model_name,
            agent_id=agent_id,
            metadata={"source": "adviser_module"},
        )

        self.advisers[adviser_id] = profile
        self._save_advisers()
        self.logger.info(
            f"Created adviser '{adviser_id}' with agent '{agent_id}' and {len(deduped_rag_collections)} RAG collections"
        )
        return adviser_id

    def update_adviser(self, adviser_id: str, req: AdviserCreateRequest) -> bool:
        """Update an existing adviser. New files are appended to the same base collection."""
        if adviser_id not in self.advisers:
            return False

        existing = self.advisers[adviser_id]

        normalized = self._normalize_prompt_and_description(
            draft_prompt=req.draft_system_prompt,
            draft_description=req.description or existing.description,
            provider_override=req.llm_provider or existing.llm_provider,
            model_override=req.model_name or existing.model_name,
        )
        system_prompt = normalized["system_prompt"]
        final_description = normalized["description"]

        provider, _, resolved_model_name = self._default_provider_and_model(
            req.llm_provider or existing.llm_provider, req.model_name or existing.model_name
        )

        # Ingest any new files into existing or new base collection
        base_collection = existing.base_collection or f"adviser_{adviser_id}_kb"
        if req.files:
            # Use helper but force the collection name to remain stable
            for file in req.files:
                try:
                    data_format = file.format
                    if isinstance(data_format, str):
                        data_format = DataFormat(data_format)
                    rag_input = RAGDataInput(
                        name=file.filename,
                        description=file.description
                        or f"Base knowledge file for adviser '{req.name}'",
                        format=data_format,
                        content=file.content,
                        tags=[adviser_id, "adviser"],
                        metadata={"adviser_id": adviser_id, "filename": file.filename},
                    )
                    success = self.rag_system.add_data_to_collection(base_collection, rag_input)
                    if not success:
                        self.logger.warning(
                            f"Failed to add data from file '{file.filename}' to collection '{base_collection}'"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error ingesting file '{file.filename}' for adviser '{adviser_id}' on update: {e}"
                    )

        rag_collections: List[str] = list(req.existing_rag_collections or existing.rag_collections)
        if base_collection:
            rag_collections.append(base_collection)
        seen: set = set()
        deduped_rag_collections: List[str] = []
        for col in rag_collections:
            if col and col not in seen:
                seen.add(col)
                deduped_rag_collections.append(col)

        # Rebuild underlying agent config and update it in AgentManager
        agent_config = self._build_agent_config(
            name=req.name,
            description=final_description,
            system_prompt=system_prompt,
            provider=provider,
            model_name=resolved_model_name,
            rag_collections=deduped_rag_collections,
        )

        agent_id = existing.agent_id
        if agent_id:
            try:
                updated = self.agent_manager.update_agent(agent_id, agent_config)
                if not updated:
                    # If update failed (e.g., id changed), recreate and overwrite
                    agent_id = self.agent_manager.create_agent(agent_config)
            except Exception as e:
                self.logger.warning(
                    f"Agent update failed for adviser '{adviser_id}', recreating agent. Error: {e}"
                )
                agent_id = self.agent_manager.create_agent(agent_config)
        else:
            agent_id = self.agent_manager.create_agent(agent_config)

        profile = AdviserProfile(
            id=adviser_id,
            name=req.name,
            description=final_description,
            system_prompt=system_prompt,
            rag_collections=deduped_rag_collections,
            base_collection=base_collection,
            llm_provider=provider,
            model_name=resolved_model_name,
            agent_id=agent_id,
            metadata=existing.metadata or {"source": "adviser_module"},
        )

        self.advisers[adviser_id] = profile
        self._save_advisers()
        self.logger.info(
            f"Updated adviser '{adviser_id}' with agent '{agent_id}' and {len(deduped_rag_collections)} RAG collections"
        )
        return True

