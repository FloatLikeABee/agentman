import logging
from typing import List, Dict, Any

from .rag_system import RAGSystem
from .llm_factory import LLMFactory, LLMProvider
from .config import settings
from .models import HelpRequest, HelpResponse, HelpSource


class HelpService:
    """The Help: AI assistant for explaining how this system works.

    - Uses a dedicated RAG collection (default: 'system_help') containing docs about modules and workflows.
    - Retrieves top-N chunks and feeds them into an LLM with a focused prompt.
    """

    def __init__(self, rag_system: RAGSystem):
        self.logger = logging.getLogger(__name__)
        self.rag_system = rag_system

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
            collection = req.rag_collection or "system_help"

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

