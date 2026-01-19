"""
Flow Step Executors - Handles execution of individual flow steps
"""
import logging
import asyncio
import json
from typing import Optional, Dict, Any, Union

from .models import FlowStepConfig, FlowStepType


class FlowStepExecutors:
    """Handles execution of different types of flow steps"""

    def __init__(
        self,
        customization_manager=None,
        agent_manager=None,
        db_tools_manager=None,
        request_tools_manager=None,
        crawler_service=None,
        rag_system=None,
        dialogue_manager=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.customization_manager = customization_manager
        self.agent_manager = agent_manager
        self.db_tools_manager = db_tools_manager
        self.request_tools_manager = request_tools_manager
        self.crawler_service = crawler_service
        self.rag_system = rag_system
        self.dialogue_manager = dialogue_manager

    async def execute_customization_step(
        self, step: FlowStepConfig, step_input: Optional[Union[str, Dict[str, Any]]], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a customization step."""
        print(f"[FLOW STEP EXECUTOR] Executing CUSTOMIZATION step: {step.step_id}")
        print(f"[FLOW STEP EXECUTOR] Input type: {type(step_input)}, Input preview: {str(step_input)[:200] if step_input else 'None'}")
        
        if not self.customization_manager:
            raise ValueError("Customization manager not available")

        profile = self.customization_manager.get_profile(step.resource_id)
        if not profile:
            raise ValueError(f"Customization profile {step.resource_id} not found")

        # Convert step_input to query string
        query = ""
        if isinstance(step_input, dict):
            # If it's a dict, try to extract a meaningful query
            query = step_input.get("response", step_input.get("output", str(step_input)))
        elif step_input:
            query = str(step_input)
        else:
            query = ""

        print(f"[FLOW STEP EXECUTOR] Extracted query: {query[:200]}...")

        # Execute customization similar to API endpoint
        from .config import settings
        from .llm_factory import LLMFactory, LLMProvider
        from .llm_langchain_wrapper import LangChainLLMWrapper
        from .models import LLMProviderType

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
        else:
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = profile.model_name or settings.gemini_default_model

        print(f"[FLOW STEP EXECUTOR] Using provider: {provider.value}, model: {model_name}")

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

        # Build context from RAG collection if specified
        rag_context = ""
        if profile.rag_collection and self.rag_system:
            results = self.rag_system.query_collection(
                profile.rag_collection,
                query,
                n_results=3,
            )
            if results:
                rag_context = "\n\n".join(r["content"] for r in results[:3])
                print(f"[FLOW STEP EXECUTOR] RAG context retrieved: {len(rag_context)} chars")

        # Build final prompt
        system_prompt = profile.system_prompt
        
        # Prepend flow context if available
        flow_context_formatted = ""
        if context and "flow_context_formatted" in context:
            flow_context_formatted = context["flow_context_formatted"]
            print(f"[FLOW STEP EXECUTOR] 📋 CUSTOMIZATION step using flow context ({len(flow_context_formatted)} chars)")
        else:
            print(f"[FLOW STEP EXECUTOR] ℹ️  CUSTOMIZATION step has no flow context (first step or not in flow)")
        
        # Build prompt with flow context, RAG context, and user query
        prompt_parts = []
        
        # Start with flow context if available
        if flow_context_formatted:
            prompt_parts.append(flow_context_formatted)
            print(f"[FLOW STEP EXECUTOR] ✅ Prepended flow context to system prompt")
        
        # Add system prompt
        prompt_parts.append(system_prompt)
        
        # Add RAG context if available
        if rag_context:
            prompt_parts.append(f"Context (from knowledge base '{profile.rag_collection}'):\n{rag_context}")
        
        # Add user query
        prompt_parts.append(f"User query:\n{query}")
        
        full_prompt = "\n\n".join(prompt_parts)

        # Direct LLM call
        response_text = await llm.ainvoke(full_prompt)
        print(f"[FLOW STEP EXECUTOR] CUSTOMIZATION step output: {response_text[:200]}...")
        return response_text

    async def execute_agent_step(
        self,
        step: FlowStepConfig,
        step_input: Optional[Union[str, Dict[str, Any]]],
        context: Dict[str, Any],
    ) -> str:
        """Execute an agent step."""
        print(f"[FLOW STEP EXECUTOR] Executing AGENT step: {step.step_id}")
        print(f"[FLOW STEP EXECUTOR] Input type: {type(step_input)}, Input preview: {str(step_input)[:200] if step_input else 'None'}")
        
        if not self.agent_manager:
            raise ValueError("Agent manager not available")

        agent_data = self.agent_manager.get_agent(step.resource_id)
        if not agent_data:
            raise ValueError(f"Agent {step.resource_id} not found")

        # Convert step_input to query string
        query = ""
        if isinstance(step_input, dict):
            query = step_input.get("response", step_input.get("output", str(step_input)))
        elif step_input:
            query = str(step_input)
        else:
            query = ""

        print(f"[FLOW STEP EXECUTOR] Extracted query: {query[:200]}...")

        # Check for flow context
        if context and "flow_context_formatted" in context:
            flow_context_formatted = context.get("flow_context_formatted", "")
            if flow_context_formatted:
                print(f"[FLOW STEP EXECUTOR] 📋 AGENT step using flow context ({len(flow_context_formatted)} chars)")
                print(f"[FLOW STEP EXECUTOR] ✅ Flow context will be prepended to agent query")
            else:
                print(f"[FLOW STEP EXECUTOR] ℹ️  AGENT step has no flow context (first step or not in flow)")
        else:
            print(f"[FLOW STEP EXECUTOR] ℹ️  AGENT step has no flow context (not in flow execution)")

        # Check if we need to inject data into system prompt
        agent_config = agent_data.get("config")
        if agent_config and hasattr(agent_config, "system_prompt_data") and step_input:
            # This will be handled by the agent_manager when it creates/updates the agent
            # For now, we'll pass it in context
            context["system_prompt_data"] = step_input

        result = await self.agent_manager.run_agent(step.resource_id, query, context)
        output = result.get("response", "")
        print(f"[FLOW STEP EXECUTOR] AGENT step output: {output[:200]}...")
        return output

    async def execute_db_tool_step(
        self, step: FlowStepConfig, step_input: Optional[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute a database tool step with optional dynamic SQL input."""
        print(f"[FLOW STEP EXECUTOR] Executing DB_TOOL step: {step.step_id}")
        print(f"[FLOW STEP EXECUTOR] Input type: {type(step_input)}, Input preview: {str(step_input)[:200] if step_input else 'None'}")
        
        if not self.db_tools_manager:
            raise ValueError("Database tools manager not available")

        profile = self.db_tools_manager.get_profile(step.resource_id)
        if not profile:
            raise ValueError(f"Database tool {step.resource_id} not found")

        # Extract SQL input from step_input
        sql_input = None
        if step_input:
            if isinstance(step_input, str):
                # Check if it looks like JSON (starts with { or [)
                step_input_stripped = step_input.strip()
                if step_input_stripped.startswith(("{", "[")):
                    self.logger.warning(
                        f"[DB TOOL STEP {step.step_id}] Received JSON string instead of SQL. "
                        f"First 100 chars: {step_input[:100]}"
                    )
                    # Don't use JSON as SQL - this would cause SQL syntax errors
                    sql_input = None
                else:
                    sql_input = step_input
            elif isinstance(step_input, dict):
                # Check if this is dialogue output (has conversation_history)
                if "conversation_history" in step_input:
                    # For dialogue output, ONLY use the "response" field if it looks like SQL
                    if "response" in step_input:
                        response_text = step_input["response"]
                        self.logger.info(
                            f"[DB TOOL STEP {step.step_id}] Detected dialogue output. "
                            f"Response type: {type(response_text)}, "
                            f"First 100 chars: {str(response_text)[:100] if response_text else 'None'}"
                        )
                        # Check if response looks like SQL (not JSON)
                        if isinstance(response_text, str):
                            response_stripped = response_text.strip()
                            if response_stripped.startswith(("{", "[")):
                                # Looks like JSON, not SQL - log warning and don't use it
                                self.logger.warning(
                                    f"[DB TOOL STEP {step.step_id}] Dialogue response appears to be JSON, not SQL. "
                                    f"Not using as SQL input. Response: {response_text[:200]}"
                                )
                                sql_input = None
                            else:
                                # Looks like SQL, use it
                                sql_input = response_text
                        elif isinstance(response_text, (dict, list)):
                            # Response is JSON data, not SQL
                            self.logger.warning(
                                f"[DB TOOL STEP {step.step_id}] Dialogue response is JSON data (dict/list), not SQL. "
                                f"Not using as SQL input."
                            )
                            sql_input = None
                        else:
                            # Convert to string and check
                            response_str = str(response_text)
                            if response_str.strip().startswith(("{", "[")):
                                self.logger.warning(
                                    f"[DB TOOL STEP {step.step_id}] Dialogue response appears to be JSON when converted to string. "
                                    f"Not using as SQL input."
                                )
                                sql_input = None
                            else:
                                sql_input = response_str
                    else:
                        self.logger.warning(
                            f"[DB TOOL STEP {step.step_id}] Dialogue output has no 'response' field. "
                            f"Available keys: {list(step_input.keys())}"
                        )
                        sql_input = None
                else:
                    # Not dialogue output, try to extract SQL from common fields
                    sql_input = step_input.get("sql", step_input.get("sql_input", step_input.get("query")))
                    if sql_input and not isinstance(sql_input, str):
                        # If it's not a string, check if it's JSON
                        if isinstance(sql_input, (dict, list)):
                            self.logger.warning(
                                f"[DB TOOL STEP {step.step_id}] Extracted SQL input is JSON data (dict/list), not SQL string. "
                                f"Not using as SQL input."
                            )
                            sql_input = None
                        else:
                            # Convert to string and check
                            sql_str = str(sql_input)
                            if sql_str.strip().startswith(("{", "[")):
                                self.logger.warning(
                                    f"[DB TOOL STEP {step.step_id}] Extracted SQL input appears to be JSON when converted to string. "
                                    f"Not using as SQL input."
                                )
                                sql_input = None
                            else:
                                sql_input = sql_str
        
        print(f"[FLOW STEP EXECUTOR] SQL input: {sql_input[:200] if sql_input else 'None'}...")
        
        # Use execute_query method which handles dynamic SQL
        # Wrap in asyncio.to_thread to ensure proper sequential execution
        result = await asyncio.to_thread(
            self.db_tools_manager.execute_query,
            tool_id=step.resource_id,
            sql_input=sql_input,
            force_refresh=True  # Always refresh for flow execution
        )

        print(f"[FLOW STEP EXECUTOR] DB_TOOL step output keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        if isinstance(result, dict):
            print(f"[FLOW STEP EXECUTOR] DB_TOOL step output preview: {json.dumps(result, indent=2)[:500]}...")
        return result

    async def execute_request_step(
        self, step: FlowStepConfig, step_input: Optional[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute a request step."""
        print(f"[FLOW STEP EXECUTOR] Executing REQUEST step: {step.step_id}")
        print(f"[FLOW STEP EXECUTOR] Input type: {type(step_input)}, Input preview: {str(step_input)[:200] if step_input else 'None'}")
        
        if not self.request_tools_manager:
            raise ValueError("Request tools manager not available")

        profile = self.request_tools_manager.get_profile(step.resource_id)
        if not profile:
            raise ValueError(f"Request profile {step.resource_id} not found")

        # Log what we received
        self.logger.info(
            f"[REQUEST STEP {step.step_id}] Received step_input. "
            f"Type: {type(step_input)}, "
            f"Keys: {list(step_input.keys()) if isinstance(step_input, dict) else 'N/A'}"
        )

        # If step_input is provided, temporarily update the profile
        original_body = None
        original_params = None
        try:
            if step_input:
                original_body = profile.body
                original_params = profile.params
                
                if isinstance(step_input, dict):
                    # Check if this is dialogue output (has conversation_history)
                    if "conversation_history" in step_input:
                        self.logger.info(
                            f"[REQUEST STEP {step.step_id}] Detected dialogue output. "
                            f"Available keys: {list(step_input.keys())}"
                        )
                        # For dialogue output, ONLY use the "response" field
                        if "response" in step_input:
                            response_text = step_input["response"]
                            self.logger.info(
                                f"[REQUEST STEP {step.step_id}] Using response from dialogue: {response_text[:100] if isinstance(response_text, str) else response_text}"
                            )
                            # Parse response_text as JSON and use it as params (replace existing params)
                            try:
                                if isinstance(response_text, str):
                                    # Try to parse as JSON
                                    parsed_params = json.loads(response_text)
                                    if isinstance(parsed_params, dict):
                                        # Use the parsed JSON dict as params (replace existing)
                                        profile.params = parsed_params
                                    else:
                                        # If parsed but not a dict, wrap in query key
                                        profile.params = {"query": parsed_params}
                                elif isinstance(response_text, dict):
                                    # Already a dict, use it directly as params (replace existing)
                                    profile.params = response_text
                                else:
                                    # Not a string or dict, convert to string and wrap in query
                                    profile.params = {"query": str(response_text)}
                            except json.JSONDecodeError:
                                # Not valid JSON, use as string in query key
                                profile.params = {"query": str(response_text)}
                        else:
                            # If no response field, log warning and use empty params
                            self.logger.warning(
                                f"[REQUEST STEP {step.step_id}] Dialogue output has no 'response' field. "
                                f"Available keys: {list(step_input.keys())}"
                            )
                            profile.params = {}
                    # Check if step_input has "query" or "body" keys
                    elif "query" in step_input:
                        # Use "query" as query parameters
                        query_data = step_input["query"]
                        if isinstance(query_data, dict):
                            profile.params = {**(profile.params or {}), **query_data}
                        else:
                            # If query is not a dict, merge with existing params
                            profile.params = profile.params or {}
                            profile.params.update({"query": query_data})
                    elif "body" in step_input:
                        # Use "body" as request body
                        profile.body = step_input["body"]
                    else:
                        # If no "query" or "body" specified, use the entire dict as query parameters
                        profile.params = {**(profile.params or {}), **step_input}
                else:
                    # If step_input is a string, use it as body
                    profile.body = str(step_input)
                
                # Update in manager
                self.request_tools_manager.requests[profile.id] = profile
                
                # Log final params and body being used
                self.logger.info(
                    f"[REQUEST STEP {step.step_id}] Final profile params: {profile.params}, "
                    f"body: {profile.body[:100] if profile.body and isinstance(profile.body, str) else profile.body}"
                )
                print(f"[FLOW STEP EXECUTOR] Final params: {json.dumps(profile.params, indent=2) if profile.params else 'None'}")
                print(f"[FLOW STEP EXECUTOR] Final body: {str(profile.body)[:200] if profile.body else 'None'}...")
            
            # Execute the request (takes request_id)
            # Wrap in asyncio.to_thread to ensure proper sequential execution
            result = await asyncio.to_thread(
                self.request_tools_manager.execute_request,
                profile.id
            )
            
            self.logger.info(
                f"[REQUEST STEP {step.step_id}] Request executed. Success: {result.get('success')}, "
                f"Status: {result.get('status_code')}"
            )
            print(f"[FLOW STEP EXECUTOR] REQUEST step output - Success: {result.get('success')}, Status: {result.get('status_code')}")
            print(f"[FLOW STEP EXECUTOR] REQUEST step output preview: {json.dumps(result, indent=2)[:500]}...")
        finally:
            # Restore original values if we modified them
            if original_body is not None:
                profile.body = original_body
            if original_params is not None:
                profile.params = original_params
            if original_body is not None or original_params is not None:
                self.request_tools_manager.requests[profile.id] = profile

        return result

    async def execute_crawler_step(
        self, step: FlowStepConfig, step_input: Optional[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute a crawler step."""
        print(f"[FLOW STEP EXECUTOR] Executing CRAWLER step: {step.step_id}")
        print(f"[FLOW STEP EXECUTOR] Input type: {type(step_input)}, Input preview: {str(step_input)[:200] if step_input else 'None'}")
        
        if not self.crawler_service:
            raise ValueError("Crawler service not available")

        # step_input should be a URL string or a dict with url
        url = ""
        if isinstance(step_input, dict):
            url = step_input.get("url", step_input.get("response", ""))
        elif step_input:
            url = str(step_input)
        else:
            raise ValueError("Crawler step requires a URL in step_input")

        print(f"[FLOW STEP EXECUTOR] Extracted URL: {url}")

        # Extract additional parameters from step metadata
        use_js = step.metadata.get("use_js", False)
        llm_provider = step.metadata.get("llm_provider")
        model = step.metadata.get("model")
        collection_name = step.metadata.get("collection_name")
        collection_description = step.metadata.get("collection_description")
        follow_links = step.metadata.get("follow_links", False)
        max_depth = step.metadata.get("max_depth", 3)
        max_pages = step.metadata.get("max_pages", 50)
        same_domain_only = step.metadata.get("same_domain_only", True)
        headers = step.metadata.get("headers")

        print(f"[FLOW STEP EXECUTOR] Crawler params - use_js: {use_js}, provider: {llm_provider}, model: {model}, follow_links: {follow_links}")

        # Wrap in asyncio.to_thread to ensure proper sequential execution
        result = await asyncio.to_thread(
            self.crawler_service.crawl_and_save,
            url=url,
            use_js=use_js,
            llm_provider=llm_provider,
            model=model,
            collection_name=collection_name,
            collection_description=collection_description,
            follow_links=follow_links,
            max_depth=max_depth,
            max_pages=max_pages,
            same_domain_only=same_domain_only,
            headers=headers,
        )

        print(f"[FLOW STEP EXECUTOR] CRAWLER step output keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        if isinstance(result, dict):
            print(f"[FLOW STEP EXECUTOR] CRAWLER step output preview: {json.dumps(result, indent=2)[:500]}...")
        return result

    async def execute_dialogue_step(
        self,
        step: FlowStepConfig,
        step_input: Optional[Union[str, Dict[str, Any]]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a dialogue step."""
        print(f"[FLOW STEP EXECUTOR] Executing DIALOGUE step: {step.step_id}")
        print(f"[FLOW STEP EXECUTOR] Input type: {type(step_input)}, Input preview: {str(step_input)[:200] if step_input else 'None'}")
        
        if not self.dialogue_manager:
            raise ValueError("Dialogue manager not available")

        profile = self.dialogue_manager.get_profile(step.resource_id)
        if not profile:
            raise ValueError(f"Dialogue profile {step.resource_id} not found")

        # Convert step_input to initial message string
        # Try step_input first, then fall back to step.input_query
        initial_message = None
        if isinstance(step_input, dict):
            # Extract message from dict
            initial_message = step_input.get("response", step_input.get("output", step_input.get("message", str(step_input))))
        elif step_input:
            initial_message = str(step_input)
        elif step.input_query:
            # Fall back to step's input_query field if step_input is not provided
            initial_message = step.input_query
        
        print(f"[FLOW STEP EXECUTOR] Initial message: {initial_message[:200] if initial_message else 'None'}...")
        
        # If no initial message is provided, return a special result indicating conversation window should open
        # The user will provide the first message through the UI
        if not initial_message or not initial_message.strip():
            result = {
                "conversation_id": None,
                "turn_number": 0,
                "max_turns": profile.max_turns,
                "response": "",
                "needs_more_info": True,
                "is_complete": False,
                "profile_id": profile.id,
                "profile_name": profile.name,
                "model_used": profile.model_name or "unknown",
                "rag_collection_used": profile.rag_collection,
                "conversation_history": [],
                "metadata": {
                    "waiting_for_initial_message": True
                }
            }
            print(f"[FLOW STEP EXECUTOR] DIALOGUE step output - waiting for initial message")
            return result

        # Check if we're continuing an existing conversation or starting new
        # IMPORTANT: Only use conversation_id from context if it belongs to THIS dialogue profile
        # This prevents different dialogue steps from reusing each other's conversations
        conversation_id = None
        context_conversation_id = context.get("conversation_id")
        
        if context_conversation_id and self.dialogue_manager:
            # Verify that the conversation belongs to this dialogue profile
            conversation = self.dialogue_manager.get_conversation(context_conversation_id)
            if conversation:
                # Check if the conversation's profile_id matches this step's resource_id
                conversation_profile_id = conversation.get("profile_id")
                if conversation_profile_id == step.resource_id:
                    # This conversation belongs to this dialogue profile - we can use it
                    conversation_id = context_conversation_id
                    self.logger.info(
                        f"[DIALOGUE STEP {step.step_id}] Using existing conversation {conversation_id} "
                        f"for dialogue profile {step.resource_id}"
                    )
                else:
                    # Conversation belongs to a different dialogue profile - start new conversation
                    self.logger.info(
                        f"[DIALOGUE STEP {step.step_id}] Context has conversation_id {context_conversation_id} "
                        f"but it belongs to profile {conversation_profile_id}, not {step.resource_id}. "
                        f"Starting new conversation."
                    )
                    conversation_id = None
        
        # Import dialogue methods from flow_dialogue_methods
        from .flow_dialogue_methods import FlowDialogueMethods
        
        dialogue_methods = FlowDialogueMethods(
            dialogue_manager=self.dialogue_manager,
            rag_system=self.rag_system
        )
        
        # Extract flow context if available
        flow_context_formatted = context.get("flow_context_formatted", "") if context else ""
        
        if flow_context_formatted:
            print(f"[FLOW STEP EXECUTOR] 📋 DIALOGUE step using flow context ({len(flow_context_formatted)} chars)")
            print(f"[FLOW STEP EXECUTOR] ✅ Flow context will be prepended to dialogue system prompt")
        else:
            print(f"[FLOW STEP EXECUTOR] ℹ️  DIALOGUE step has no flow context (first step or not in flow)")
        
        if conversation_id:
            # Check if conversation is already complete
            if self.dialogue_manager:
                conversation = self.dialogue_manager.get_conversation(conversation_id)
                if conversation:
                    turn_number = conversation.get("turn_number", 0)
                    max_turns = conversation.get("max_turns", 999)
                    messages = conversation.get("messages", [])
                    
                    # If conversation reached max turns, return the final result
                    if turn_number >= max_turns:
                        profile = self.dialogue_manager.get_profile(step.resource_id)
                        if profile:
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
                            
                            result = {
                                "conversation_id": conversation_id,
                                "turn_number": turn_number,
                                "max_turns": max_turns,
                                "response": final_response,
                                "needs_more_info": False,
                                "is_complete": True,
                                "profile_id": step.resource_id,
                                "profile_name": profile.name,
                                "model_used": profile.model_name or "unknown",
                                "rag_collection_used": profile.rag_collection,
                                "conversation_history": [
                                    msg.model_dump() if hasattr(msg, 'model_dump') else msg 
                                    for msg in messages
                                ],
                                "metadata": {}
                            }
                            print(f"[FLOW STEP EXECUTOR] DIALOGUE step output - conversation complete, response: {final_response[:200]}...")
                            return result
            
            # Continue existing conversation (not complete yet)
            from .models import DialogueContinueRequest
            request = DialogueContinueRequest(
                user_message=initial_message,
                conversation_id=conversation_id
            )
            result = await dialogue_methods.continue_dialogue_internal(
                step.resource_id, request, flow_context_formatted=flow_context_formatted
            )
        else:
            # Start new conversation
            from .models import DialogueStartRequest
            request = DialogueStartRequest(
                initial_message=initial_message,
                n_results=3,
            )
            result = await dialogue_methods.start_dialogue_internal(
                step.resource_id, request, flow_context_formatted=flow_context_formatted
            )
            # Store conversation_id in context for next steps
            if result and "conversation_id" in result:
                context["conversation_id"] = result["conversation_id"]

        print(f"[FLOW STEP EXECUTOR] DIALOGUE step output - conversation_id: {result.get('conversation_id')}, "
              f"is_complete: {result.get('is_complete')}, response: {result.get('response', '')[:200]}...")
        
        # Include dialogue response data in the result for frontend to display conversation UI
        # The result is a DialogueResponse, which contains conversation_id, conversation_history, etc.
        return result

