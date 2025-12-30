"""
Flow Service - Orchestrates workflows combining Customization, Agents, DBTools, Requests, and Crawler
"""
import logging
import os
import time
import asyncio
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from tinydb import TinyDB, Query

from .config import settings
from .models import (
    FlowProfile,
    FlowCreateRequest,
    FlowUpdateRequest,
    FlowExecuteRequest,
    FlowExecuteResponse,
    FlowStepResult,
    FlowStepType,
    FlowStepConfig,
)


class FlowService:
    """Manage and execute workflow flows"""

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
        self.flows: Dict[str, FlowProfile] = {}
        self.customization_manager = customization_manager
        self.agent_manager = agent_manager
        self.db_tools_manager = db_tools_manager
        self.request_tools_manager = request_tools_manager
        self.crawler_service = crawler_service
        self.rag_system = rag_system
        self.dialogue_manager = dialogue_manager

        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "flows.json")
        self.db = TinyDB(self.db_path)
        self.query = Query()

        self._load_flows()

    def _load_flows(self) -> None:
        """Load flow profiles from TinyDB."""
        try:
            docs = self.db.all()
            for doc in docs:
                try:
                    flow_id = doc.get("id")
                    data = doc.get("profile", {})
                    profile = FlowProfile(id=flow_id, **data)
                    self.flows[flow_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load flow {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.flows)} flow profiles")
        except Exception as e:
            self.logger.error(f"Error loading flows: {e}")

    def _save_flows(self) -> None:
        """Persist all flow profiles to TinyDB."""
        try:
            self.db.truncate()
            for flow_id, profile in self.flows.items():
                self.db.insert(
                    {
                        "id": flow_id,
                        "profile": profile.model_dump(exclude={"id"}),
                    }
                )
            self.logger.info(f"Saved {len(self.flows)} flow profiles")
        except Exception as e:
            self.logger.error(f"Error saving flows: {e}")

    def _generate_id(self, name: str) -> str:
        """Generate a stable, URL-friendly id from the flow name."""
        base_id = name.strip().lower().replace(" ", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        if not base_id:
            base_id = "flow"

        candidate = base_id
        counter = 1
        while candidate in self.flows:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def list_flows(self) -> List[FlowProfile]:
        """List all flow profiles."""
        return list(self.flows.values())

    def get_flow(self, flow_id: str) -> Optional[FlowProfile]:
        """Get a flow profile by ID."""
        return self.flows.get(flow_id)

    def create_flow(self, req: FlowCreateRequest) -> str:
        """Create a new flow profile."""
        flow_id = self._generate_id(req.name)
        now = datetime.now().isoformat()
        profile = FlowProfile(
            id=flow_id,
            name=req.name,
            description=req.description,
            steps=req.steps,
            is_active=req.is_active,
            created_at=now,
            updated_at=now,
            metadata=req.metadata or {},
        )
        self.flows[flow_id] = profile
        self._save_flows()
        self.logger.info(f"Created flow profile: {flow_id}")
        return flow_id

    def update_flow(self, flow_id: str, req: FlowUpdateRequest) -> bool:
        """Update an existing flow profile."""
        if flow_id not in self.flows:
            return False
        try:
            existing = self.flows[flow_id]
            profile = FlowProfile(
                id=flow_id,
                name=req.name,
                description=req.description,
                steps=req.steps,
                is_active=req.is_active,
                created_at=existing.created_at,
                updated_at=datetime.now().isoformat(),
                metadata=req.metadata or {},
            )
            self.flows[flow_id] = profile
            self._save_flows()
            self.logger.info(f"Updated flow profile: {flow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating flow {flow_id}: {e}")
            return False

    def delete_flow(self, flow_id: str) -> bool:
        """Delete a flow profile."""
        if flow_id not in self.flows:
            return False
        try:
            del self.flows[flow_id]
            self.db.remove(self.query.id == flow_id)
            self.logger.info(f"Deleted flow profile: {flow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting flow {flow_id}: {e}")
            return False

    async def execute_flow(
        self, flow_id: str, request: FlowExecuteRequest
    ) -> FlowExecuteResponse:
        """
        Execute a flow and return results from all steps.
        
        Args:
            flow_id: Flow ID to execute
            request: Execution request with initial input and context
            resume_from_step: Optional step index (1-based) to resume from (for paused flows)
            previous_step_results: Optional previous step results when resuming
        """
        if flow_id not in self.flows:
            raise ValueError(f"Flow {flow_id} not found")

        flow = self.flows[flow_id]
        if not flow.is_active:
            raise ValueError(f"Flow {flow_id} is not active")

        start_time = time.time()
        step_results: List[FlowStepResult] = request.previous_step_results.copy() if request.previous_step_results else []
        previous_output: Optional[Union[str, Dict[str, Any]]] = None
        
        # If resuming, we need to re-execute the dialogue step to get the final result
        # The dialogue step was paused, but now the conversation is complete
        if request.resume_from_step and step_results:
            # Get the step we're resuming from (the step before resume_from_step)
            resume_step_index = request.resume_from_step - 1
            if resume_step_index > 0 and resume_step_index <= len(flow.steps):
                resume_step = flow.steps[resume_step_index - 1]  # Convert to 0-based index
                
                # If the previous step was a dialogue step, re-execute it to get final result
                if resume_step.step_type == FlowStepType.DIALOGUE:
                    self.logger.info(
                        f"[FLOW {flow_id}] Re-executing dialogue step {resume_step.step_id} to get final result"
                    )
                    try:
                        # Get conversation_id from context
                        conversation_id = request.context.get("conversation_id") if request.context else None
                        if conversation_id:
                            # Re-execute the dialogue step with the conversation context
                            # This will return the final result since conversation is complete
                            dialogue_output = await self._execute_dialogue_step(
                                resume_step,
                                "",  # Empty initial message since conversation already started
                                request.context or {}
                            )
                            
                            # Update the step result with the final output
                            if step_results and len(step_results) > 0:
                                last_result = step_results[-1]
                                if last_result.step_id == resume_step.step_id:
                                    # Update the existing step result
                                    last_result.output = dialogue_output
                                    last_result.metadata = {
                                        "dialogue_response": {
                                            "conversation_id": dialogue_output.get("conversation_id"),
                                            "conversation_history": dialogue_output.get("conversation_history", []),
                                            "response": dialogue_output.get("response"),
                                            "is_complete": dialogue_output.get("is_complete", True),
                                            "needs_more_info": False,
                                            "turn_number": dialogue_output.get("turn_number"),
                                            "max_turns": dialogue_output.get("max_turns"),
                                            "profile_id": dialogue_output.get("profile_id"),
                                            "profile_name": dialogue_output.get("profile_name"),
                                            "metadata": dialogue_output.get("metadata", {}),
                                        }
                                    }
                            
                            # Set previous_output to the final dialogue result
                            previous_output = dialogue_output
                            self.logger.info(
                                f"[FLOW {flow_id}] Dialogue step {resume_step.step_id} final result obtained. "
                                f"Output type: {type(previous_output)}"
                            )
                        else:
                            # No conversation_id, use previous step result
                            if step_results:
                                last_result = step_results[-1]
                                if last_result.success and last_result.output:
                                    previous_output = last_result.output
                    except Exception as e:
                        self.logger.warning(
                            f"[FLOW {flow_id}] Error re-executing dialogue step: {e}. "
                            f"Using previous step result output."
                        )
                        # Fallback to using the previous step result
                        if step_results:
                            last_result = step_results[-1]
                            if last_result.success and last_result.output:
                                previous_output = last_result.output
                    except Exception as e:
                        self.logger.error(f"Error processing dialogue step on resume: {e}")
                        # Fallback to using the previous step result
                        if step_results:
                            last_result = step_results[-1]
                            if last_result.success and last_result.output:
                                previous_output = last_result.output
                else:
                    # Not a dialogue step, just use the previous output
                    last_result = step_results[-1]
                    if last_result.success and last_result.output:
                        previous_output = last_result.output
            else:
                # Fallback: use last result output
                if step_results:
                    last_result = step_results[-1]
                    if last_result.success and last_result.output:
                        previous_output = last_result.output

        try:
            # Start from resume step if provided, otherwise start from beginning
            start_index = request.resume_from_step if request.resume_from_step else 1
            for step_index, step in enumerate(flow.steps, 1):
                # Skip steps before resume point
                if step_index < start_index:
                    continue
                step_start = time.time()
                self.logger.info(
                    f"[FLOW {flow_id}] Starting step {step_index}/{len(flow.steps)}: {step.step_id} ({step.step_type})"
                )

                try:
                    # Determine input for this step
                    step_input = None
                    if step.use_previous_output and previous_output is not None:
                        # Use previous step output
                        if isinstance(previous_output, dict):
                            # Map previous output to step input based on output_mapping
                            if step.output_mapping:
                                step_input = {}
                                for key, source in step.output_mapping.items():
                                    if source in previous_output:
                                        step_input[key] = previous_output[source]
                            else:
                                # Use entire previous output as input
                                step_input = previous_output
                        else:
                            # Previous output is a string
                            step_input = str(previous_output)
                    elif step.input_query:
                        # Use explicit input query
                        step_input = step.input_query
                    elif request.initial_input and len(step_results) == 0:
                        # Use initial input for first step
                        step_input = request.initial_input
                    else:
                        step_input = None

                    # Execute step based on type - ensure sequential execution
                    self.logger.info(
                        f"[FLOW {flow_id}] Executing step {step_index}/{len(flow.steps)}: {step.step_id} - waiting for completion..."
                    )
                    output = await self._execute_step(
                        step, step_input, request.context or {}
                    )
                    # Ensure step is fully complete before proceeding
                    self.logger.info(
                        f"[FLOW {flow_id}] Step {step_index}/{len(flow.steps)}: {step.step_id} completed successfully"
                    )

                    execution_time = time.time() - step_start
                    
                    # For dialogue steps, include dialogue response data in metadata
                    metadata = {}
                    dialogue_waiting = False
                    if step.step_type == FlowStepType.DIALOGUE and isinstance(output, dict):
                        # Include dialogue response information for frontend to display conversation UI
                        # Also include the metadata from the output if it exists (e.g., waiting_for_initial_message)
                        dialogue_metadata = output.get("metadata", {})
                        is_complete = output.get("is_complete", False)
                        waiting_for_initial = dialogue_metadata.get("waiting_for_initial_message", False)
                        needs_more_info = output.get("needs_more_info", False)
                        conversation_id = output.get("conversation_id")
                        
                        # Only block if waiting for initial message (no conversation started yet)
                        # OR if conversation exists but is not complete AND we need more info
                        # But if conversation is complete, don't block
                        if waiting_for_initial:
                            # No conversation started yet - definitely block
                            dialogue_waiting = True
                        elif conversation_id and not is_complete and needs_more_info:
                            # Conversation exists but not complete and needs more info
                            # Check actual conversation state from dialogue manager
                            if self.dialogue_manager:
                                try:
                                    conversation = self.dialogue_manager.get_conversation(conversation_id)
                                    if conversation:
                                        # Check if conversation is actually complete
                                        actual_is_complete = (
                                            conversation.get("turn_number", 0) >= conversation.get("max_turns", 999)
                                        )
                                        if not actual_is_complete:
                                            # Still waiting for more input
                                            dialogue_waiting = True
                                        else:
                                            # Conversation reached max turns, consider it complete
                                            dialogue_waiting = False
                                    else:
                                        # Conversation not found, don't block
                                        dialogue_waiting = False
                                except Exception as e:
                                    self.logger.warning(f"Error checking conversation state: {e}")
                                    # If we can't check, don't block to avoid infinite blocking
                                    dialogue_waiting = False
                            else:
                                # Can't check, don't block
                                dialogue_waiting = False
                        else:
                            # Conversation is complete or no waiting needed
                            dialogue_waiting = False
                        
                        metadata = {
                            "dialogue_response": {
                                "conversation_id": conversation_id,
                                "conversation_history": output.get("conversation_history", []),
                                "response": output.get("response"),
                                "is_complete": is_complete,
                                "needs_more_info": needs_more_info,
                                "turn_number": output.get("turn_number"),
                                "max_turns": output.get("max_turns"),
                                "profile_id": output.get("profile_id"),
                                "profile_name": output.get("profile_name"),
                                "metadata": dialogue_metadata,  # Include nested metadata
                            }
                        }
                        
                        if dialogue_waiting:
                            self.logger.info(
                                f"[FLOW {flow_id}] Dialogue step {step.step_id} is waiting for user input. "
                                f"Flow execution paused at step {step_index}/{len(flow.steps)}."
                            )
                        else:
                            self.logger.info(
                                f"[FLOW {flow_id}] Dialogue step {step.step_id} is complete or ready to proceed. "
                                f"is_complete={is_complete}, conversation_id={conversation_id}"
                            )
                    
                    step_result = FlowStepResult(
                        step_id=step.step_id,
                        step_name=step.step_name,
                        step_type=step.step_type,
                        success=True,
                        output=output,
                        error=None,
                        execution_time=execution_time,
                        metadata=metadata,
                    )
                    step_results.append(step_result)
                    previous_output = output
                    
                    # If dialogue step is waiting for user input, return immediately to avoid blocking UI
                    # The frontend will handle the dialogue interaction and can resume the flow when complete
                    if dialogue_waiting:
                        self.logger.info(
                            f"[FLOW {flow_id}] Flow execution paused - dialogue step {step.step_id} waiting for user input. "
                            f"Returning control to frontend."
                        )
                        total_time = time.time() - start_time
                        return FlowExecuteResponse(
                            flow_id=flow_id,
                            flow_name=flow.name,
                            success=False,  # Not fully successful yet
                            step_results=step_results,
                            final_output=previous_output,
                            total_execution_time=total_time,
                            error=f"Flow paused at step {step_index} ({step.step_id}): Dialogue waiting for user input",
                            metadata={
                                "paused": True,
                                "paused_at_step": step_index,
                                "paused_step_id": step.step_id,
                                "waiting_for_dialogue": True,
                                "dialogue_conversation_id": conversation_id,
                                "dialogue_profile_id": output.get("profile_id"),
                                "can_resume": True,  # Indicates flow can be resumed
                            },
                        )

                except Exception as e:
                    execution_time = time.time() - step_start
                    error_msg = str(e)
                    self.logger.error(
                        f"Error executing step {step.step_id}: {error_msg}", exc_info=True
                    )
                    step_result = FlowStepResult(
                        step_id=step.step_id,
                        step_name=step.step_name,
                        step_type=step.step_type,
                        success=False,
                        output=None,
                        error=error_msg,
                        execution_time=execution_time,
                        metadata={},
                    )
                    step_results.append(step_result)
                    # Stop execution on error
                    break

            total_time = time.time() - start_time
            final_output = previous_output if step_results and step_results[-1].success else None
            success = all(r.success for r in step_results)

            return FlowExecuteResponse(
                flow_id=flow_id,
                flow_name=flow.name,
                success=success,
                step_results=step_results,
                final_output=final_output,
                total_execution_time=total_time,
                error=None if success else "One or more steps failed",
                metadata={},
            )

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Error executing flow {flow_id}: {error_msg}", exc_info=True)
            return FlowExecuteResponse(
                flow_id=flow_id,
                flow_name=flow.name,
                success=False,
                step_results=step_results,
                final_output=None,
                total_execution_time=total_time,
                error=error_msg,
                metadata={},
            )

    async def _execute_step(
        self,
        step: FlowStepConfig,
        step_input: Optional[Union[str, Dict[str, Any]]],
        context: Dict[str, Any],
    ) -> Union[str, Dict[str, Any]]:
        """Execute a single flow step."""
        if step.step_type == FlowStepType.CUSTOMIZATION:
            return await self._execute_customization_step(step, step_input)

        elif step.step_type == FlowStepType.AGENT:
            return await self._execute_agent_step(step, step_input, context)

        elif step.step_type == FlowStepType.DB_TOOL:
            return await self._execute_db_tool_step(step, step_input)

        elif step.step_type == FlowStepType.REQUEST:
            return await self._execute_request_step(step, step_input)

        elif step.step_type == FlowStepType.CRAWLER:
            return await self._execute_crawler_step(step, step_input)

        elif step.step_type == FlowStepType.DIALOGUE:
            return await self._execute_dialogue_step(step, step_input, context)

        else:
            raise ValueError(f"Unknown step type: {step.step_type}")

    async def _execute_customization_step(
        self, step: FlowStepConfig, step_input: Optional[Union[str, Dict[str, Any]]]
    ) -> str:
        """Execute a customization step."""
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
        context = ""
        if profile.rag_collection and self.rag_system:
            results = self.rag_system.query_collection(
                profile.rag_collection,
                query,
                n_results=3,
            )
            if results:
                context = "\n\n".join(r["content"] for r in results[:3])

        # Build final prompt
        system_prompt = profile.system_prompt
        if context:
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Context (from knowledge base '{profile.rag_collection}'):\n{context}\n\n"
                f"User query:\n{query}"
            )
        else:
            full_prompt = f"{system_prompt}\n\nUser query:\n{query}"

        # Direct LLM call
        response_text = await llm.ainvoke(full_prompt)
        return response_text

    async def _execute_agent_step(
        self,
        step: FlowStepConfig,
        step_input: Optional[Union[str, Dict[str, Any]]],
        context: Dict[str, Any],
    ) -> str:
        """Execute an agent step."""
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

        # Check if we need to inject data into system prompt
        agent_config = agent_data.get("config")
        if agent_config and hasattr(agent_config, "system_prompt_data") and step_input:
            # This will be handled by the agent_manager when it creates/updates the agent
            # For now, we'll pass it in context
            context["system_prompt_data"] = step_input

        result = await self.agent_manager.run_agent(step.resource_id, query, context)
        return result.get("response", "")

    async def _execute_db_tool_step(
        self, step: FlowStepConfig, step_input: Optional[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute a database tool step with optional dynamic SQL input."""
        if not self.db_tools_manager:
            raise ValueError("Database tools manager not available")

        profile = self.db_tools_manager.get_profile(step.resource_id)
        if not profile:
            raise ValueError(f"Database tool {step.resource_id} not found")

        # Extract SQL input from step_input
        sql_input = None
        if step_input:
            if isinstance(step_input, str):
                sql_input = step_input
            elif isinstance(step_input, dict):
                # If it's a dict, try to extract SQL from common fields
                sql_input = step_input.get("sql", step_input.get("sql_input", step_input.get("query")))
                if sql_input and not isinstance(sql_input, str):
                    # If it's not a string, try to convert it
                    sql_input = str(sql_input)
        
        # Use execute_query method which handles dynamic SQL
        # Wrap in asyncio.to_thread to ensure proper sequential execution
        result = await asyncio.to_thread(
            self.db_tools_manager.execute_query,
            tool_id=step.resource_id,
            sql_input=sql_input,
            force_refresh=True  # Always refresh for flow execution
        )

        return result

    async def _execute_request_step(
        self, step: FlowStepConfig, step_input: Optional[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute a request step."""
        if not self.request_tools_manager:
            raise ValueError("Request tools manager not available")

        profile = self.request_tools_manager.get_profile(step.resource_id)
        if not profile:
            raise ValueError(f"Request profile {step.resource_id} not found")

        # If step_input is provided, temporarily update the profile
        original_body = None
        original_params = None
        try:
            if step_input:
                original_body = profile.body
                original_params = profile.params
                
                if isinstance(step_input, dict):
                    # Check if step_input has "query" or "body" keys
                    if "query" in step_input:
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
            
            # Execute the request (takes request_id)
            # Wrap in asyncio.to_thread to ensure proper sequential execution
            result = await asyncio.to_thread(
                self.request_tools_manager.execute_request,
                profile.id
            )
        finally:
            # Restore original values if we modified them
            if original_body is not None:
                profile.body = original_body
            if original_params is not None:
                profile.params = original_params
            if original_body is not None or original_params is not None:
                self.request_tools_manager.requests[profile.id] = profile

        return result

    async def _execute_crawler_step(
        self, step: FlowStepConfig, step_input: Optional[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute a crawler step."""
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

        # Extract additional parameters from step metadata
        use_js = step.metadata.get("use_js", False)
        llm_provider = step.metadata.get("llm_provider")
        model = step.metadata.get("model")
        collection_name = step.metadata.get("collection_name")
        collection_description = step.metadata.get("collection_description")

        # Wrap in asyncio.to_thread to ensure proper sequential execution
        result = await asyncio.to_thread(
            self.crawler_service.crawl_and_save,
            url=url,
            use_js=use_js,
            llm_provider=llm_provider,
            model=model,
            collection_name=collection_name,
            collection_description=collection_description,
        )

        return result

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
        
        # If no initial message is provided, return a special result indicating conversation window should open
        # The user will provide the first message through the UI
        if not initial_message or not initial_message.strip():
            return {
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

        # Check if we're continuing an existing conversation or starting new
        conversation_id = context.get("conversation_id")
        
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
                            
                            return {
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
            
            # Continue existing conversation (not complete yet)
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

        # Include dialogue response data in the result for frontend to display conversation UI
        # The result is a DialogueResponse, which contains conversation_id, conversation_history, etc.
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

    async def _wait_for_dialogue_completion(
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
                # Try to get the current conversation state even if not complete
                if conversation_id:
                    try:
                        conversation = self.dialogue_manager.get_conversation(conversation_id)
                        if conversation:
                            profile = self.dialogue_manager.get_profile(dialogue_id)
                            return {
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
                            
                            return {
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
                                break
                except Exception as e:
                    self.logger.debug(f"Error checking for new conversations: {e}")
            
            # Log status periodically
            if time.time() - last_log_time >= log_interval:
                self.logger.info(
                    f"[FLOW {flow_id}] Still waiting for dialogue step {step_id} "
                    f"(elapsed: {elapsed:.0f}s / {timeout_seconds}s)"
                )
                last_log_time = time.time()
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)

