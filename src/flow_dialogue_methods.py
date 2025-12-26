    async def _execute_dialogue_step(
        self,
        step: FlowStepConfig,
        step_input: Optional[Union[str, Dict[str, Any]]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a dialogue step."""
        if not self.dialogue_manager:
            raise ValueError("Dialogue manager not available")

        profile = self.dialogue_manager.get_profile(step.resource_id)
        if not profile:
            raise ValueError(f"Dialogue profile {step.resource_id} not found")

        # Convert step_input to initial message string
        initial_message = ""
        if isinstance(step_input, dict):
            # Extract message from dict
            initial_message = step_input.get("response", step_input.get("output", step_input.get("message", str(step_input))))
        elif step_input:
            initial_message = str(step_input)
        else:
            raise ValueError("Dialogue step requires an initial message in step_input")

        # Check if we're continuing an existing conversation or starting new
        conversation_id = context.get("conversation_id")
        
        if conversation_id:
            # Continue existing conversation
            from .models import DialogueContinueRequest
            request = DialogueContinueRequest(
                user_message=initial_message,
                conversation_id=conversation_id
            )
            result = await self._continue_dialogue_internal(step.resource_id, request)
        else:
            # Start new conversation
            from .models import DialogueStartRequest
            request = DialogueStartRequest(
                initial_message=initial_message,
                n_results=3,
            )
            result = await self._start_dialogue_internal(step.resource_id, request)
            # Store conversation_id in context for next steps
            if result and "conversation_id" in result:
                context["conversation_id"] = result["conversation_id"]

        return result

    async def _start_dialogue_internal(
        self,
        dialogue_id: str,
        request: "DialogueStartRequest"
    ) -> Dict[str, Any]:
        """Internal method to start a dialogue conversation (replicates API logic)"""
        from .config import settings
        from .llm_factory import LLMFactory, LLMProvider
        from .llm_langchain_wrapper import LangChainLLMWrapper
        from .models import LLMProviderType

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
        else:
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = profile.model_name or settings.gemini_default_model

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

        # Create conversation
        conversation_id = self.dialogue_manager._create_conversation(
            dialogue_id,
            request.initial_message,
            turn_number=1
        )

        # Build prompt
        system_prompt = profile.system_prompt
        if context:
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Context (from knowledge base '{rag_used}'):\n{context}\n\n"
                f"User message:\n{request.initial_message}"
            )
        else:
            full_prompt = f"{system_prompt}\n\nUser message:\n{request.initial_message}"

        # Call LLM
        response_text = await llm.ainvoke(full_prompt)

        # Add assistant response to conversation
        self.dialogue_manager._add_message_to_conversation(
            conversation_id,
            "assistant",
            response_text
        )

        # Determine if more info is needed (simple heuristic: check for question marks)
        needs_more_info = "?" in response_text or "please" in response_text.lower() or "could you" in response_text.lower()
        is_complete = not needs_more_info or self.dialogue_manager.active_conversations[conversation_id]["turn_number"] >= profile.max_turns

        # Get conversation history
        conversation = self.dialogue_manager.get_conversation(conversation_id)
        conversation_history = conversation["messages"] if conversation else []

        return {
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

    async def _continue_dialogue_internal(
        self,
        dialogue_id: str,
        request: "DialogueContinueRequest"
    ) -> Dict[str, Any]:
        """Internal method to continue a dialogue conversation (replicates API logic)"""
        from .config import settings
        from .llm_factory import LLMFactory, LLMProvider
        from .llm_langchain_wrapper import LangChainLLMWrapper
        from .models import LLMProviderType

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
        else:
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = profile.model_name or settings.gemini_default_model

        # Create LLM caller
        llm_caller = LLMFactory.create_caller(
            provider=LLMProvider(provider.value),
            api_key=api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=8192,
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

        # Build conversation history string
        conversation_history_str = "\n".join([
            f"{msg.role}: {msg.content if hasattr(msg, 'content') else msg.get('content', '')}"
            for msg in conversation["messages"]
        ])

        # Build prompt
        system_prompt = profile.system_prompt
        if context:
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Context (from knowledge base '{rag_used}'):\n{context}\n\n"
                f"Conversation history:\n{conversation_history_str}\n\n"
                f"User message:\n{request.user_message}"
            )
        else:
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation history:\n{conversation_history_str}\n\n"
                f"User message:\n{request.user_message}"
            )

        # Call LLM
        response_text = await llm.ainvoke(full_prompt)

        # Add assistant response
        self.dialogue_manager._add_message_to_conversation(
            request.conversation_id,
            "assistant",
            response_text
        )

        # Increment turn
        self.dialogue_manager._increment_turn(request.conversation_id)
        conversation = self.dialogue_manager.get_conversation(request.conversation_id)

        # Determine completion status
        needs_more_info = "?" in response_text or "please" in response_text.lower() or "could you" in response_text.lower()
        is_complete = not needs_more_info or conversation["turn_number"] >= conversation["max_turns"]

        # Get updated conversation history
        conversation_history = conversation["messages"] if conversation else []

        return {
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

