import logging
import os
from typing import List, Dict, Any

from .rag_system import RAGSystem
from .llm_factory import LLMFactory, LLMProvider
from .config import settings
from .models import HelpRequest, HelpResponse, HelpSource, RAGDataInput, DataFormat


SYSTEM_HELP_COLLECTION = "system_help"


class HelpService:
    """The Help: AI assistant for explaining how this system works.

    - Uses a dedicated RAG collection (default: 'system_help') containing docs about modules and workflows.
    - Retrieves top-N chunks and feeds them into an LLM with a focused prompt.
    """

    def __init__(self, rag_system: RAGSystem):
        self.logger = logging.getLogger(__name__)
        self.rag_system = rag_system
        # Auto-bootstrap documentation collection so The Help works out of the box.
        self.ensure_system_help_collection()

    def _build_system_help_document(self) -> str:
        """Build the base documentation content for The Help collection."""
        sections = [
            "# Ground Control - System Help Knowledge Base",
            "",
            "## Purpose",
            "This document explains how the Ground Control system modules work and how they connect.",
            "",
            "## Core Modules",
            "- RAG Manager: Create/query collections and ingest documents.",
            "- Agent Manager: Create agents with provider/model, tools, and optional RAG collections.",
            "- Tool Manager: Configure built-in tools and custom behavior.",
            "- DB Tools: Configure database profiles and run queries/Text-to-SQL.",
            "- Request Tools: Configure external/internal HTTP requests.",
            "- Dialogues / Conversations: Structured multi-turn AI workflows.",
            "- Flows: Chain resources (agent/db/request/dialogue) into automation steps.",
            "- MCP Hosts: Configure external MCP server endpoints and transports.",
            "- System: Monitor status and manage system settings.",
            "",
            "## Key Concepts",
            "- RAG collections are knowledge bases used during retrieval.",
            "- Agents can use tools and/or RAG collections depending on configuration.",
            "- Text-to-SQL can use DB Tool profiles, connection configs, or connection strings.",
            "- The Help assistant uses the `system_help` RAG collection to answer questions.",
            "",
            "## Best Practices",
            "- Keep collection names stable and descriptive.",
            "- Validate data before ingestion.",
            "- Use read-only DB users for DB tools whenever possible.",
            "- Keep API keys out of source code and store in local env/key files.",
            "",
            "## Notes",
            "- This collection is protected and reserved for The Help assistant.",
            "- Add/refresh docs through backend bootstrap processes only.",
        ]

        # Include README content for broader system explanations.
        readme_text = ""
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            readme_path = os.path.join(root_dir, "README.md")
            if os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_text = f.read().strip()
        except Exception as e:
            self.logger.warning(f"Failed to read README for system help bootstrap: {e}")

        if readme_text:
            sections.extend(
                [
                    "",
                    "## README Reference",
                    "The following content is imported from project README for detailed usage/reference:",
                    "",
                    readme_text,
                ]
            )
        return "\n".join(sections)

    def ensure_system_help_collection(self) -> None:
        """Create and seed `system_help` collection if missing or empty."""
        try:
            collections = self.rag_system.list_collections()
            existing = next((c for c in collections if c.get("name") == SYSTEM_HELP_COLLECTION), None)
            if existing and int(existing.get("count", 0) or 0) > 0:
                return

            doc_text = self._build_system_help_document()
            data_input = RAGDataInput(
                name="system_help_bootstrap",
                description="Protected system help documentation for The Help assistant",
                format=DataFormat.TXT,
                content=doc_text,
                tags=["system", "help", "documentation", "protected"],
                metadata={
                    "protected": True,
                    "source": "bootstrap",
                    "module": "help_service",
                },
            )
            ok = self.rag_system.add_data_to_collection(SYSTEM_HELP_COLLECTION, data_input)
            if ok:
                self.logger.info("Initialized system_help collection with bootstrap documentation.")
            else:
                self.logger.warning("Failed to initialize system_help bootstrap documentation.")
        except Exception as e:
            self.logger.error(f"Error ensuring system_help collection: {e}")

    def _get_llm(self):
        """Create an LLM caller with default provider/model."""
        provider_str = getattr(settings, "default_llm_provider", "gemini").lower()
        model_name = getattr(settings, "default_model", None)

        if provider_str == "gemini":
            provider = LLMProvider.GEMINI
            api_key = settings.gemini_api_key
            model = model_name or settings.gemini_default_model
        elif provider_str == "qwen":
            provider = LLMProvider.QWEN
            api_key = settings.qwen_api_key
            model = model_name or settings.qwen_default_model
        elif provider_str == "mistral":
            provider = LLMProvider.MISTRAL
            api_key = settings.mistral_api_key
            model = model_name or settings.mistral_default_model
        elif provider_str == "groq":
            provider = LLMProvider.GROQ
            api_key = settings.groq_api_key
            model = model_name or settings.groq_default_model
        else:
            provider = LLMProvider.GEMINI
            api_key = settings.gemini_api_key
            model = settings.gemini_default_model

        return LLMFactory.create_caller(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=0.3,
            max_tokens=2048,
        )

    def ask(self, req: HelpRequest) -> HelpResponse:
        """Answer a help question about the system using RAG + LLM."""
        try:
            collection = req.rag_collection or SYSTEM_HELP_COLLECTION

            # Check that the collection exists
            collections = [c["name"] for c in self.rag_system.list_collections()]
            if collection not in collections:
                return HelpResponse(
                    answer=(
                        "The system help knowledge base is not initialized yet.\n\n"
                        f"Please create a RAG collection named `{collection}` and add documentation "
                        "about your modules, workflows, and usage. Once populated, The Help will "
                        "be able to answer questions about how your system works."
                    ),
                    sources=[],
                    used_rag=False,
                    error="system_help_collection_missing",
                )

            # Query RAG
            results = self.rag_system.query_collection(collection, req.question, req.n_results)

            if not results:
                context_text = "No documentation was found in the system_help collection."
                used_rag = False
            else:
                used_rag = True
                # Concatenate top chunks as context
                context_parts: List[str] = []
                sources: List[HelpSource] = []
                for r in results:
                    text = r.get("content") or r.get("text") or ""
                    meta: Dict[str, Any] = r.get("metadata") or {}
                    score = r.get("score") or meta.get("score")
                    doc_id = meta.get("id") or meta.get("document_id")
                    context_parts.append(text)
                    sources.append(
                        HelpSource(
                            collection=collection,
                            document_id=doc_id,
                            score=score,
                            metadata=meta,
                        )
                    )
                context_text = "\n\n---\n\n".join(context_parts)

            llm = self._get_llm()

            prompt = (
                "You are **The Help**, an AI assistant whose only job is to explain how this system works.\n\n"
                "You have access to internal documentation about modules, APIs, flows, and UI behavior.\n"
                "Always answer in **clear, structured markdown**, with headings and bullet points where helpful.\n\n"
                "### Documentation context\n\n"
                f"{context_text}\n\n"
                "### User question\n\n"
                f"{req.question}\n\n"
                "### Instructions\n"
                "- Base your answer on the documentation context whenever possible.\n"
                "- If something is not covered, say so explicitly and respond with best-effort guidance.\n"
                "- Prefer concrete steps and references to module names / screens / endpoints.\n"
            )

            answer = llm.generate(prompt)

            # Build sources list again (so we always return it)
            sources_out: List[HelpSource] = []
            if used_rag and results:
                for r in results:
                    meta: Dict[str, Any] = r.get("metadata") or {}
                    score = r.get("score") or meta.get("score")
                    doc_id = meta.get("id") or meta.get("document_id")
                    sources_out.append(
                        HelpSource(
                            collection=collection,
                            document_id=doc_id,
                            score=score,
                            metadata=meta,
                        )
                    )

            return HelpResponse(
                answer=answer,
                sources=sources_out,
                used_rag=used_rag,
                error=None,
            )
        except Exception as e:
            self.logger.error(f"Error in HelpService.ask: {e}")
            return HelpResponse(
                answer="Sorry, The Help encountered an error while trying to answer your question.",
                sources=[],
                used_rag=False,
                error=str(e),
            )

