"""
Flow Dialogue Methods - Internal dialogue conversation management
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any

from .config import settings
from .llm_factory import LLMFactory, LLMProvider
from .llm_langchain_wrapper import LangChainLLMWrapper
from .models import LLMProviderType


class FlowDialogueMethods:
    """Internal methods for managing dialogue conversations within flows"""

    def __init__(self, dialogue_manager=None, rag_system=None):
        self.logger = logging.getLogger(__name__)
        self.dialogue_manager = dialogue_manager
        self.rag_system = rag_system

    async def start_dialogue_internal(
        self,
        dialogue_id: str,
        request: "DialogueStartRequest",
        flow_context_formatted: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal method to start a dialogue conversation (replicates API logic)"""
        print(f"[FLOW DIALOGUE] Starting dialogue: {dialogue_id}")
        print(f"[FLOW DIALOGUE] Initial message: {request.initial_message[:200] if request.initial_message else 'None'}...")
        
        profile = self.dialogue_manager.get_profile(dialogue_id)
        if not profile:
            raise ValueError(f"Dialogue profile {dialogue_id} not found")

        # Determine provider/model
        provider_str = (
            profile.llm_provider.value
            if profile.llm_provider
            else settings.default_llm_provider
        )
        if provider_str == "gemini":
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = profile.model_name or settings.gemini_default_model
        elif provider_str == "qwen":
            provider = LLMProviderType.QWEN
            api_key = settings.qwen_api_key
            model_name = profile.model_name or settings.qwen_default_model
        elif provider_str == "mistral":
            provider = LLMProviderType.MISTRAL
            api_key = getattr(settings, 'mistral_api_key', '')
            model_name = profile.model_name or "mistral-large-latest"
        elif provider_str == "groq":
            provider = LLMProviderType.GROQ
            api_key = getattr(settings, "groq_api_key", "")
            model_name = profile.model_name or getattr(settings, "groq_default_model", "llama-3.3-70b-versatile")
        else:
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = profile.model_name or settings.gemini_default_model

        print(f"[FLOW DIALOGUE] Using provider: {provider.value}, model: {model_name}")

        # Create LLM caller
        llm_caller = LLMFactory.create_caller(
            provider=LLMProvider(provider.value),
            api_key=api_key,
            model=model_name,
            temperature=request.temperature if request.temperature is not None else 0.7,
            max_tokens=request.max_tokens if request.max_tokens is not None else 8192,
        )

        # Wrap in LangChain-compatible wrapper
        llm = LangChainLLMWrapper(llm_caller=llm_caller)

        # Build context from RAG collection if specified
        context = ""
        rag_used: Optional[str] = None
        if profile.rag_collection and self.rag_system:
            rag_used = profile.rag_collection
            results = self.rag_system.query_collection(
                profile.rag_collection,
                request.initial_message,
                request.n_results,
            )
            if results:
                context = "\n\n".join(r["content"] for r in results[: request.n_results])
                print(f"[FLOW DIALOGUE] RAG context retrieved: {len(context)} chars")

        # Create conversation
        conversation_id = self.dialogue_manager._create_conversation(
            dialogue_id,
            request.initial_message,
            turn_number=1
        )

        print(f"[FLOW DIALOGUE] Created conversation: {conversation_id}")

        # Build prompt with flow context, system prompt, RAG context, and user message
        system_prompt = profile.system_prompt
        prompt_parts = []
        
        # Start with flow context if available
        if flow_context_formatted:
            print(f"[FLOW DIALOGUE] 📋 Prepending flow context to dialogue system prompt ({len(flow_context_formatted)} chars)")
            prompt_parts.append(flow_context_formatted)
            print(f"[FLOW DIALOGUE] ✅ Flow context prepended to dialogue prompt")
        else:
            print(f"[FLOW DIALOGUE] ℹ️  No flow context available for dialogue start")
        
        # Add system prompt
        prompt_parts.append(system_prompt)
        
        # Add RAG context if available
        if context:
            prompt_parts.append(f"Context (from knowledge base '{rag_used}'):\n{context}")
        
        # Add user message
        prompt_parts.append(f"User message:\n{request.initial_message}")
        
        full_prompt = "\n\n".join(prompt_parts)

        # Call LLM
        response_text = await llm.ainvoke(full_prompt)
        print(f"[FLOW DIALOGUE] LLM response: {response_text[:200]}...")

        # Add assistant response to conversation
        self.dialogue_manager._add_message_to_conversation(
            conversation_id,
            "assistant",
            response_text
        )

        # Determine if more info is needed - improved logic to detect when conversation is actually complete
        response_lower = response_text.lower()
        
        # Check for completion indicators (AI is ready to proceed)
        completion_phrases = [
            "thank you", "i have all", "i have the", "i've got", "i've collected",
            "ready to", "proceed", "all set", "complete", "finished", "done",
            "i understand", "got it", "perfect", "that's all", "no more questions",
            "sufficient information", "enough information", "all the information"
        ]
        has_completion_phrase = any(phrase in response_lower for phrase in completion_phrases)
        
        # Check for asking indicators (AI needs more info)
        asking_phrases = ["?", "can you", "could you", "please provide", "i need", 
                         "what", "which", "when", "where", "how", "tell me",
                         "missing", "need more", "need additional", "require"]
        is_asking = any(phrase in response_lower for phrase in asking_phrases)
        
        # Determine if more info is needed
        # If AI has completion phrase and is not asking, it's complete
        # If AI is asking questions, it needs more info (unless we've hit max turns)
        needs_more_info = is_asking and not has_completion_phrase
        is_complete = has_completion_phrase or (not needs_more_info) or self.dialogue_manager.active_conversations[conversation_id]["turn_number"] >= profile.max_turns

        # Get conversation history
        conversation = self.dialogue_manager.get_conversation(conversation_id)
        conversation_history = conversation["messages"] if conversation else []

        result = {
            "conversation_id": conversation_id,
            "turn_number": 1,
            "max_turns": profile.max_turns,
            "response": response_text,
            "needs_more_info": needs_more_info,
            "is_complete": is_complete,
            "profile_id": dialogue_id,
            "profile_name": profile.name,
            "model_used": model_name,
            "rag_collection_used": rag_used,
            "conversation_history": [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in conversation_history],
        }
        
        print(f"[FLOW DIALOGUE] Dialogue started - conversation_id: {conversation_id}, "
              f"needs_more_info: {needs_more_info}, is_complete: {is_complete}")
        
        return result

    async def continue_dialogue_internal(
        self,
        dialogue_id: str,
        request: "DialogueContinueRequest",
        flow_context_formatted: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal method to continue a dialogue conversation (replicates API logic)"""
        print(f"[FLOW DIALOGUE] Continuing dialogue: {dialogue_id}, conversation: {request.conversation_id}")
        print(f"[FLOW DIALOGUE] User message: {request.user_message[:200] if request.user_message else 'None'}...")
        
        profile = self.dialogue_manager.get_profile(dialogue_id)
        if not profile:
            raise ValueError(f"Dialogue profile {dialogue_id} not found")

        conversation = self.dialogue_manager.get_conversation(request.conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {request.conversation_id} not found")

        if conversation["turn_number"] >= conversation["max_turns"]:
            raise ValueError(f"Maximum turns ({conversation['max_turns']}) reached for this conversation")

        # Determine provider/model
        provider_str = (
            profile.llm_provider.value
            if profile.llm_provider
            else settings.default_llm_provider
        )
        if provider_str == "gemini":
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = profile.model_name or settings.gemini_default_model
        elif provider_str == "qwen":
            provider = LLMProviderType.QWEN
            api_key = settings.qwen_api_key
            model_name = profile.model_name or settings.qwen_default_model
        elif provider_str == "mistral":
            provider = LLMProviderType.MISTRAL
            api_key = getattr(settings, 'mistral_api_key', '')
            model_name = profile.model_name or "mistral-large-latest"
        elif provider_str == "groq":
            provider = LLMProviderType.GROQ
            api_key = getattr(settings, "groq_api_key", "")
            model_name = profile.model_name or getattr(settings, "groq_default_model", "llama-3.3-70b-versatile")
        else:
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = profile.model_name or settings.gemini_default_model

        print(f"[FLOW DIALOGUE] Using provider: {provider.value}, model: {model_name}")

        # Create LLM caller
        llm_caller = LLMFactory.create_caller(
            provider=LLMProvider(provider.value),
            api_key=api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=8192,
            timeout=settings.api_timeout,
        )

        # Wrap in LangChain-compatible wrapper
        llm = LangChainLLMWrapper(llm_caller=llm_caller)

        # Add user message to conversation
        self.dialogue_manager._add_message_to_conversation(
            request.conversation_id,
            "user",
            request.user_message
        )

        # Build context from RAG if needed
        context = ""
        rag_used: Optional[str] = None
        if profile.rag_collection and self.rag_system:
            rag_used = profile.rag_collection
            # Use the latest user message for RAG query
            results = self.rag_system.query_collection(
                profile.rag_collection,
                request.user_message,
                n_results=3,
            )
            if results:
                context = "\n\n".join(r["content"] for r in results[:3])
                print(f"[FLOW DIALOGUE] RAG context retrieved: {len(context)} chars")

        # Build conversation history string
        conversation_history_str = "\n".join([
            f"{msg.role}: {msg.content if hasattr(msg, 'content') else msg.get('content', '')}"
            for msg in conversation["messages"]
        ])

        # Build prompt with flow context, system prompt, RAG context, conversation history, and user message
        system_prompt = profile.system_prompt
        prompt_parts = []
        
        # Start with flow context if available
        if flow_context_formatted:
            print(f"[FLOW DIALOGUE] 📋 Prepending flow context to dialogue system prompt ({len(flow_context_formatted)} chars)")
            prompt_parts.append(flow_context_formatted)
            print(f"[FLOW DIALOGUE] ✅ Flow context prepended to dialogue continue prompt")
        else:
            print(f"[FLOW DIALOGUE] ℹ️  No flow context available for dialogue continue")
        
        # Add system prompt
        prompt_parts.append(system_prompt)
        
        # Add RAG context if available
        if context:
            prompt_parts.append(f"Context (from knowledge base '{rag_used}'):\n{context}")
        
        # Add conversation history
        prompt_parts.append(f"Conversation history:\n{conversation_history_str}")
        
        # Add user message
        prompt_parts.append(f"User message:\n{request.user_message}")
        
        full_prompt = "\n\n".join(prompt_parts)

        # Call LLM
        response_text = await llm.ainvoke(full_prompt)
        print(f"[FLOW DIALOGUE] LLM response: {response_text[:200]}...")

        # Add assistant response
        self.dialogue_manager._add_message_to_conversation(
            request.conversation_id,
            "assistant",
            response_text
        )

        # Increment turn
        self.dialogue_manager._increment_turn(request.conversation_id)
        conversation = self.dialogue_manager.get_conversation(request.conversation_id)

        # Determine completion status - improved logic to detect when conversation is actually complete
        response_lower = response_text.lower()
        
        # Check for completion indicators (AI is ready to proceed)
        completion_phrases = [
            "thank you", "i have all", "i have the", "i've got", "i've collected",
            "ready to", "proceed", "all set", "complete", "finished", "done",
            "i understand", "got it", "perfect", "that's all", "no more questions",
            "sufficient information", "enough information", "all the information"
        ]
        has_completion_phrase = any(phrase in response_lower for phrase in completion_phrases)
        
        # Check for asking indicators (AI needs more info)
        asking_phrases = ["?", "can you", "could you", "please provide", "i need", 
                         "what", "which", "when", "where", "how", "tell me",
                         "missing", "need more", "need additional", "require"]
        is_asking = any(phrase in response_lower for phrase in asking_phrases)
        
        # Determine if more info is needed
        # If AI has completion phrase and is not asking, it's complete
        # If AI is asking questions, it needs more info (unless we've hit max turns)
        needs_more_info = is_asking and not has_completion_phrase
        is_complete = has_completion_phrase or (not needs_more_info) or conversation["turn_number"] >= conversation["max_turns"]

        # Get updated conversation history
        conversation_history = conversation["messages"] if conversation else []

        result = {
            "conversation_id": request.conversation_id,
            "turn_number": conversation["turn_number"],
            "max_turns": conversation["max_turns"],
            "response": response_text,
            "needs_more_info": needs_more_info,
            "is_complete": is_complete,
            "profile_id": dialogue_id,
            "profile_name": profile.name,
            "model_used": model_name,
            "rag_collection_used": rag_used,
            "conversation_history": [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in conversation_history],
        }
        
        print(f"[FLOW DIALOGUE] Dialogue continued - turn: {conversation['turn_number']}/{conversation['max_turns']}, "
              f"needs_more_info: {needs_more_info}, is_complete: {is_complete}")
        
        return result

    async def wait_for_dialogue_completion(
        self,
        conversation_id: Optional[str],
        dialogue_id: str,
        timeout_seconds: int = 3600,
        flow_id: str = "",
        step_id: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a dialogue conversation to complete by polling the dialogue manager.
        Returns the final dialogue result when complete, or None if timeout.
        
        Args:
            conversation_id: The conversation ID to wait for (None if waiting for initial message)
            dialogue_id: The dialogue profile ID
            timeout_seconds: Maximum time to wait (default: 1 hour)
            flow_id: Flow ID for logging
            step_id: Step ID for logging
            
        Returns:
            Final dialogue result dict if conversation completes, None if timeout
        """
        print(f"[FLOW DIALOGUE] Waiting for dialogue completion - flow: {flow_id}, step: {step_id}, "
              f"conversation: {conversation_id}, timeout: {timeout_seconds}s")
        
        if not self.dialogue_manager:
            self.logger.warning(f"[FLOW {flow_id}] Dialogue manager not available, cannot wait for completion")
            return None
        
        start_time = time.time()
        poll_interval = 2.0  # Poll every 2 seconds
        last_log_time = start_time
        log_interval = 30.0  # Log status every 30 seconds
        
        self.logger.info(
            f"[FLOW {flow_id}] Waiting for dialogue {dialogue_id} to complete "
            f"(conversation_id={conversation_id}, timeout={timeout_seconds}s)"
        )
        
        while True:
            elapsed = time.time() - start_time
            
            # Check timeout
            if elapsed >= timeout_seconds:
                self.logger.warning(
                    f"[FLOW {flow_id}] Dialogue step {step_id} timed out after {timeout_seconds}s. Proceeding anyway."
                )
                print(f"[FLOW DIALOGUE] Timeout after {elapsed:.1f}s")
                # Try to get the current conversation state even if not complete
                if conversation_id:
                    try:
                        conversation = self.dialogue_manager.get_conversation(conversation_id)
                        if conversation:
                            profile = self.dialogue_manager.get_profile(dialogue_id)
                            result = {
                                "conversation_id": conversation_id,
                                "turn_number": conversation.get("turn_number", 0),
                                "max_turns": conversation.get("max_turns", 5),
                                "response": "Dialogue timed out - proceeding with flow",
                                "needs_more_info": False,
                                "is_complete": True,  # Mark as complete to proceed
                                "profile_id": dialogue_id,
                                "profile_name": profile.name if profile else dialogue_id,
                                "model_used": "unknown",
                                "rag_collection_used": None,
                                "conversation_history": [
                                    msg.model_dump() if hasattr(msg, 'model_dump') else msg 
                                    for msg in conversation.get("messages", [])
                                ],
                                "metadata": {"timeout": True, "elapsed_seconds": elapsed}
                            }
                            print(f"[FLOW DIALOGUE] Returning timeout result")
                            return result
                    except Exception as e:
                        self.logger.error(f"Error getting conversation state on timeout: {e}")
                return None
            
            # Check if conversation exists and is complete
            if conversation_id:
                try:
                    conversation = self.dialogue_manager.get_conversation(conversation_id)
                    if conversation:
                        turn_number = conversation.get("turn_number", 0)
                        max_turns = conversation.get("max_turns", 999)
                        messages = conversation.get("messages", [])
                        
                        # Check if conversation reached max turns or has a final response
                        is_complete = turn_number >= max_turns
                        
                        # If complete, get the final dialogue result
                        if is_complete:
                            profile = self.dialogue_manager.get_profile(dialogue_id)
                            if not profile:
                                self.logger.warning(f"Dialogue profile {dialogue_id} not found")
                                return None
                            
                            # Get the last assistant message as the final response
                            last_assistant_msg = None
                            for msg in reversed(messages):
                                if hasattr(msg, 'role') and msg.role == 'assistant':
                                    last_assistant_msg = msg
                                    break
                                elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                                    last_assistant_msg = msg
                                    break
                            
                            final_response = (
                                last_assistant_msg.content if hasattr(last_assistant_msg, 'content')
                                else last_assistant_msg.get('content', '') if last_assistant_msg
                                else "Dialogue completed"
                            )
                            
                            self.logger.info(
                                f"[FLOW {flow_id}] Dialogue step {step_id} completed after {elapsed:.1f}s "
                                f"(turn {turn_number}/{max_turns})"
                            )
                            
                            result = {
                                "conversation_id": conversation_id,
                                "turn_number": turn_number,
                                "max_turns": max_turns,
                                "response": final_response,
                                "needs_more_info": False,
                                "is_complete": True,
                                "profile_id": dialogue_id,
                                "profile_name": profile.name,
                                "model_used": profile.model_name or "unknown",
                                "rag_collection_used": profile.rag_collection,
                                "conversation_history": [
                                    msg.model_dump() if hasattr(msg, 'model_dump') else msg 
                                    for msg in messages
                                ],
                                "metadata": {"elapsed_seconds": elapsed}
                            }
                            
                            print(f"[FLOW DIALOGUE] Dialogue completed after {elapsed:.1f}s - "
                                  f"turn {turn_number}/{max_turns}, response: {final_response[:200]}...")
                            
                            return result
                    else:
                        # Conversation not found - might have been deleted or never created
                        # If we've waited a bit, assume it's not coming and proceed
                        if elapsed > 60:  # Wait at least 1 minute before giving up
                            self.logger.warning(
                                f"[FLOW {flow_id}] Conversation {conversation_id} not found after {elapsed:.1f}s. Proceeding."
                            )
                            return None
                except Exception as e:
                    self.logger.error(f"Error checking conversation state: {e}")
            else:
                # No conversation_id yet - waiting for initial message
                # Check if a conversation was created for this dialogue_id
                # We need to find any active conversation for this dialogue profile
                try:
                    # Get all active conversations and find one for this dialogue_id
                    # Note: This is a workaround since we don't have a direct way to list conversations by dialogue_id
                    # We'll check if any conversation exists and matches
                    if hasattr(self.dialogue_manager, 'active_conversations'):
                        for conv_id, conv in self.dialogue_manager.active_conversations.items():
                            if conv.get("profile_id") == dialogue_id:
                                # Found a conversation for this dialogue - update conversation_id and continue checking
                                conversation_id = conv_id
                                self.logger.info(
                                    f"[FLOW {flow_id}] Found conversation {conversation_id} for dialogue {dialogue_id}"
                                )
                                print(f"[FLOW DIALOGUE] Found conversation: {conversation_id}")
                                break
                except Exception as e:
                    self.logger.debug(f"Error checking for new conversations: {e}")
            
            # Log status periodically
            if time.time() - last_log_time >= log_interval:
                self.logger.info(
                    f"[FLOW {flow_id}] Still waiting for dialogue step {step_id} "
                    f"(elapsed: {elapsed:.0f}s / {timeout_seconds}s)"
                )
                print(f"[FLOW DIALOGUE] Still waiting... elapsed: {elapsed:.0f}s / {timeout_seconds}s")
                last_log_time = time.time()
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
