"""
Special Flow Service - Dialogue-Driven Flow management and execution
"""
import logging
import os
import time
import asyncio
import json
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from tinydb import TinyDB, Query

from .config import settings
from .models import (
    SpecialFlow1Profile,
    SpecialFlow1CreateRequest,
    SpecialFlow1UpdateRequest,
    SpecialFlow1ExecuteRequest,
    SpecialFlow1ExecuteResponse,
)


class SpecialFlow1Service:
    """Service for managing and executing Dialogue-Driven Flow"""

    def __init__(
        self,
        db_tools_manager=None,
        request_tools_manager=None,
        dialogue_manager=None,
        rag_system=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.flows: Dict[str, SpecialFlow1Profile] = {}
        self.db_tools_manager = db_tools_manager
        self.request_tools_manager = request_tools_manager
        self.dialogue_manager = dialogue_manager
        self.rag_system = rag_system

        os.makedirs(settings.data_directory, exist_ok=True)
        self.db_path = os.path.join(settings.data_directory, "special_flows_1.json")
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
                    profile = SpecialFlow1Profile(id=flow_id, **data)
                    self.flows[flow_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load flow {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.flows)} dialogue-driven flow profiles")
            print(f"[SPECIAL FLOW SERVICE] Loaded {len(self.flows)} dialogue-driven flow profiles")
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
            self.logger.info(f"Saved {len(self.flows)} dialogue-driven flow profiles")
            print(f"[SPECIAL FLOW SERVICE] Saved {len(self.flows)} dialogue-driven flow profiles")
        except Exception as e:
            self.logger.error(f"Error saving flows: {e}")

    def _generate_id(self, name: str) -> str:
        """Generate a stable, URL-friendly id from the flow name."""
        base_id = name.strip().lower().replace(" ", "_")
        base_id = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        if not base_id:
            base_id = "special_flow_1"

        candidate = base_id
        counter = 1
        while candidate in self.flows:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def list_flows(self) -> List[SpecialFlow1Profile]:
        """List all flow profiles."""
        print(f"[SPECIAL FLOW SERVICE] Listing {len(self.flows)} flows")
        return list(self.flows.values())

    def get_flow(self, flow_id: str) -> Optional[SpecialFlow1Profile]:
        """Get a flow profile by ID."""
        result = self.flows.get(flow_id)
        print(f"[SPECIAL FLOW SERVICE] Get flow {flow_id}: {'found' if result else 'not found'}")
        return result

    def create_flow(self, req: SpecialFlow1CreateRequest) -> str:
        """Create a new flow profile."""
        print(f"[SPECIAL FLOW SERVICE] Creating special flow: {req.name}")
        
        flow_id = self._generate_id(req.name)
        now = datetime.now().isoformat()
        profile = SpecialFlow1Profile(
            id=flow_id,
            name=req.name,
            description=req.description,
            config=req.config,
            is_active=req.is_active,
            created_at=now,
            updated_at=now,
            metadata=req.metadata or {},
        )
        self.flows[flow_id] = profile
        self._save_flows()
        self.logger.info(f"Created dialogue-driven flow profile: {flow_id}")
        print(f"[SPECIAL FLOW SERVICE] Created flow profile: {flow_id}")
        return flow_id

    def update_flow(self, flow_id: str, req: SpecialFlow1UpdateRequest) -> bool:
        """Update an existing flow profile."""
        print(f"[SPECIAL FLOW SERVICE] Updating flow: {flow_id}")
        
        if flow_id not in self.flows:
            print(f"[SPECIAL FLOW SERVICE] Flow {flow_id} not found")
            return False
        try:
            existing = self.flows[flow_id]
            profile = SpecialFlow1Profile(
                id=flow_id,
                name=req.name,
                description=req.description,
                config=req.config,
                is_active=req.is_active,
                created_at=existing.created_at,
                updated_at=datetime.now().isoformat(),
                metadata=req.metadata or {},
            )
            self.flows[flow_id] = profile
            self._save_flows()
            self.logger.info(f"Updated dialogue-driven flow profile: {flow_id}")
            print(f"[SPECIAL FLOW SERVICE] Updated flow profile: {flow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating flow {flow_id}: {e}")
            print(f"[SPECIAL FLOW SERVICE] Error updating flow {flow_id}: {e}")
            return False

    def delete_flow(self, flow_id: str) -> bool:
        """Delete a flow profile."""
        print(f"[SPECIAL FLOW SERVICE] Deleting flow: {flow_id}")
        
        if flow_id not in self.flows:
            print(f"[SPECIAL FLOW SERVICE] Flow {flow_id} not found")
            return False
        try:
            del self.flows[flow_id]
            self.db.remove(self.query.id == flow_id)
            self.logger.info(f"Deleted dialogue-driven flow profile: {flow_id}")
            print(f"[SPECIAL FLOW SERVICE] Deleted flow profile: {flow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting flow {flow_id}: {e}")
            print(f"[SPECIAL FLOW SERVICE] Error deleting flow {flow_id}: {e}")
            return False

    async def execute_flow(
        self, flow_id: str, request: SpecialFlow1ExecuteRequest
    ) -> SpecialFlow1ExecuteResponse:
        """
        Execute a Dialogue-Driven Flow.
        
        Steps:
        1. Fetch initial data (DB tool or Request tool)
        2. Start dialogue with initial data (caches all conversation)
        3. Fetch data after dialogue (Request tool, uses cached conversation)
        4. Final processing with all data (uses cached conversation)
        5. Call final API
        """
        print(f"[SPECIAL FLOW SERVICE] ========== EXECUTING DIALOGUE-DRIVEN FLOW: {flow_id} ==========")
        print(f"[SPECIAL FLOW SERVICE] Initial input: {str(request.initial_input)[:200] if request.initial_input else 'None'}...")
        print(f"[SPECIAL FLOW SERVICE] Resume from phase: {request.resume_from_phase if request.resume_from_phase else 'None'}")
        
        if flow_id not in self.flows:
            raise ValueError(f"Flow {flow_id} not found")

        flow: SpecialFlow1Profile = self.flows[flow_id]
        if not flow.is_active:
            raise ValueError(f"Flow {flow_id} is not active")

        start_time = time.time()
        config = flow.config

        try:
            # Check if resuming from a specific phase
            resume_from = request.resume_from_phase
            dialogue_phase1_result = request.dialogue_phase1_result
            initial_data = request.initial_data
            
            if resume_from == "dialogue_phase1" or resume_from == "dialogue":
                if not dialogue_phase1_result:
                    raise ValueError(f"dialogue_phase1_result is required when resuming from {resume_from}")
                if not initial_data:
                    raise ValueError("initial_data is required when resuming")
                # Check if dialogue is complete
                if dialogue_phase1_result.get("needs_user_input") and not dialogue_phase1_result.get("is_complete"):
                    raise ValueError("Cannot resume: dialogue is still waiting for user input and is not complete")
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== RESUMING FROM DIALOGUE ==========")
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Dialogue result: conversation_id={dialogue_phase1_result.get('conversation_id')}, is_complete={dialogue_phase1_result.get('is_complete')}")
                print(f"[SPECIAL FLOW SERVICE] Resuming from dialogue phase - conversation_id: {dialogue_phase1_result.get('conversation_id')}")
                # Cache the dialogue conversation for the entire session - step 3 and step 4 will use this cached conversation
                # The conversation_history contains the full dialogue outcome from step 2, which is defined by the dialogue prompt
                current_dialogue_result = dialogue_phase1_result
                conversation_id = current_dialogue_result.get("conversation_id")
                dialogue_id = current_dialogue_result.get("dialogue_id")
                
                # Log that conversation is cached for later steps
                if dialogue_phase1_result.get('conversation_history'):
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Using cached conversation from previous session - {len(dialogue_phase1_result.get('conversation_history', []))} messages available for steps 3 and 4")
                    print(f"[SPECIAL FLOW SERVICE] Using cached conversation - {len(dialogue_phase1_result.get('conversation_history', []))} messages")
            else:
                # Step 1: Fetch initial data
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== STEP 1: Fetching initial data ==========")
                print(f"[SPECIAL FLOW SERVICE] ========== STEP 1: Fetching initial data ==========")
                initial_data = None
                if config.initial_data_source.type == "db_tool":
                    if not self.db_tools_manager:
                        raise ValueError("Database tools manager not available")
                    sql_input = request.initial_input if request.initial_input else config.initial_data_source.sql_input
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Using DB tool: {config.initial_data_source.resource_id}, SQL: {sql_input}")
                    print(f"[SPECIAL FLOW SERVICE] Using DB tool: {config.initial_data_source.resource_id}, SQL: {sql_input[:200] if sql_input else 'None'}...")
                    initial_data = await asyncio.to_thread(
                        self.db_tools_manager.execute_query,
                        tool_id=config.initial_data_source.resource_id,
                        sql_input=sql_input,
                        force_refresh=True,
                    )
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 1 OUTPUT: Initial data fetched: {json.dumps(initial_data, indent=2) if initial_data else 'None'}")
                    print(f"[SPECIAL FLOW SERVICE] Step 1 OUTPUT - Initial data keys: {list(initial_data.keys()) if isinstance(initial_data, dict) else 'N/A'}")
                elif config.initial_data_source.type == "request_tool":
                    if not self.request_tools_manager:
                        raise ValueError("Request tools manager not available")
                    # If initial_input provided, use it to update request params
                    if request.initial_input:
                        profile = self.request_tools_manager.get_profile(config.initial_data_source.resource_id)
                        if profile:
                            try:
                                parsed = json.loads(request.initial_input) if isinstance(request.initial_input, str) else request.initial_input
                                if isinstance(parsed, dict):
                                    profile.params = {**(profile.params or {}), **parsed}
                                    self.request_tools_manager.requests[profile.id] = profile
                                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Updated request params: {json.dumps(parsed, indent=2)}")
                                    print(f"[SPECIAL FLOW SERVICE] Updated request params: {json.dumps(parsed, indent=2)[:200]}...")
                            except Exception as e:
                                self.logger.warning(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Failed to parse initial_input: {e}")
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Using Request tool: {config.initial_data_source.resource_id}")
                    print(f"[SPECIAL FLOW SERVICE] Using Request tool: {config.initial_data_source.resource_id}")
                    initial_data = await asyncio.to_thread(
                        self.request_tools_manager.execute_request,
                        config.initial_data_source.resource_id
                    )
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 1 OUTPUT: Initial data fetched: {json.dumps(initial_data, indent=2) if initial_data else 'None'}")
                    print(f"[SPECIAL FLOW SERVICE] Step 1 OUTPUT - Initial data keys: {list(initial_data.keys()) if isinstance(initial_data, dict) else 'N/A'}")

                # Step 2: Start dialogue (continuous - stays open until step 7)
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== STEP 2: Starting dialogue ==========")
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] System prompt: {config.dialogue_config.system_prompt[:200]}...")
                print(f"[SPECIAL FLOW SERVICE] ========== STEP 2: Starting dialogue ==========")
                print(f"[SPECIAL FLOW SERVICE] System prompt preview: {config.dialogue_config.system_prompt[:200]}...")
                dialogue_phase1_result = await self._start_dialogue_phase1(
                    flow_id, config, initial_data, request.context or {}
                )
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 2 OUTPUT: Dialogue result - conversation_id: {dialogue_phase1_result.get('conversation_id')}, needs_user_input: {dialogue_phase1_result.get('needs_user_input')}, is_complete: {dialogue_phase1_result.get('is_complete')}")
                print(f"[SPECIAL FLOW SERVICE] Step 2 OUTPUT - conversation_id: {dialogue_phase1_result.get('conversation_id')}, "
                      f"needs_user_input: {dialogue_phase1_result.get('needs_user_input')}, "
                      f"is_complete: {dialogue_phase1_result.get('is_complete')}")
                if dialogue_phase1_result.get('conversation_history'):
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Conversation history length: {len(dialogue_phase1_result.get('conversation_history', []))}")
                    print(f"[SPECIAL FLOW SERVICE] Conversation history length: {len(dialogue_phase1_result.get('conversation_history', []))}")

                # Cache dialogue conversation for the entire session - step 3 and step 4 will use this cached conversation
                # The conversation_history contains the full dialogue outcome from step 2, which is defined by the dialogue prompt
                current_dialogue_result = dialogue_phase1_result
                conversation_id = dialogue_phase1_result.get("conversation_id")
                dialogue_id = dialogue_phase1_result.get("dialogue_id")
                
                # Log that conversation is cached for later steps
                if dialogue_phase1_result.get('conversation_history'):
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Cached conversation for session - {len(dialogue_phase1_result.get('conversation_history', []))} messages will be available for steps 3 and 4")
                    print(f"[SPECIAL FLOW SERVICE] Cached conversation - {len(dialogue_phase1_result.get('conversation_history', []))} messages")
                
                # Check if we need to pause for initial user interaction
                if dialogue_phase1_result.get("needs_user_input") and not dialogue_phase1_result.get("is_complete"):
                    print(f"[SPECIAL FLOW SERVICE] Flow paused - waiting for user input in dialogue")
                    return SpecialFlow1ExecuteResponse(
                        flow_id=flow_id,
                        flow_name=flow.name,
                        success=False,
                        phase="dialogue",
                        initial_data=initial_data,
                        dialogue_phase1=dialogue_phase1_result,
                        fetched_data=None,
                        dialogue_phase2=None,
                        final_outcome=None,
                        api_call_result=None,
                        total_execution_time=time.time() - start_time,
                        error="Flow paused - waiting for user input in dialogue",
                        metadata={"paused": True, "waiting_for_dialogue": True, "conversation_id": conversation_id, "continuous_dialogue": True},
                    )

            # Step 3: Fetch data after dialogue (using the dialogue outcome from step 2)
            # The dialogue outcome (conversation_history) is defined by the dialogue prompt in step 2
            # Step 3 extracts information from the cached conversation based on the prompt's instructions
            fetched_data = None
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== STEP 3: Fetching data after dialogue ==========")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Request tool ID: {config.mid_dialogue_request.request_tool_id}")
            print(f"[SPECIAL FLOW SERVICE] ========== STEP 3: Fetching data after dialogue ==========")
            print(f"[SPECIAL FLOW SERVICE] Request tool ID: {config.mid_dialogue_request.request_tool_id}")
            fetched_data = await self._fetch_mid_dialogue_data(
                config.mid_dialogue_request,
                current_dialogue_result  # Use cached conversation from step 2 - contains dialogue outcome defined by prompt
            )
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 3 OUTPUT: Fetched data: {json.dumps(fetched_data, indent=2) if fetched_data else 'None'}")
            print(f"[SPECIAL FLOW SERVICE] Step 3 OUTPUT - Fetched data keys: {list(fetched_data.keys()) if isinstance(fetched_data, dict) else 'N/A'}")
            # Print the response body from step 3 (short preview)
            step3_response_body = None
            if isinstance(fetched_data, dict):
                step3_response_body = fetched_data.get('response_data') or fetched_data.get('response_body') or fetched_data.get('body') or fetched_data.get('data')
                if step3_response_body:
                    body_preview = json.dumps(step3_response_body, indent=2) if isinstance(step3_response_body, dict) else str(step3_response_body)
                    print(f"[SPECIAL FLOW SERVICE] Step 3 RESPONSE BODY: {body_preview[:200]}..." if len(body_preview) > 200 else f"[SPECIAL FLOW SERVICE] Step 3 RESPONSE BODY: {body_preview}")
                else:
                    # If no response_data, use the whole fetched_data
                    step3_response_body = fetched_data
                    body_preview = json.dumps(fetched_data, indent=2)
                    print(f"[SPECIAL FLOW SERVICE] Step 3 RESPONSE BODY (using full fetched_data): {body_preview[:200]}..." if len(body_preview) > 200 else f"[SPECIAL FLOW SERVICE] Step 3 RESPONSE BODY: {body_preview}")
            else:
                step3_response_body = fetched_data
                body_str = str(fetched_data)
                print(f"[SPECIAL FLOW SERVICE] Step 3 RESPONSE BODY: {body_str[:200]}..." if len(body_str) > 200 else f"[SPECIAL FLOW SERVICE] Step 3 RESPONSE BODY: {body_str}")

            # Step 4: Final processing (use cached conversation from step 2 - available throughout the session)
            # The full conversation history from step 2 is cached and used here
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== STEP 4: Final processing ==========")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] System prompt: {config.final_processing.system_prompt[:200]}...")
            print(f"[SPECIAL FLOW SERVICE] ========== STEP 4: Final processing ==========")
            print(f"[SPECIAL FLOW SERVICE] System prompt preview: {config.final_processing.system_prompt[:200]}...")
            # Extract response body from step 3 for step 4
            step3_response_body = None
            if isinstance(fetched_data, dict):
                step3_response_body = fetched_data.get('response_data') or fetched_data.get('response_body') or fetched_data.get('body') or fetched_data.get('data') or fetched_data
            
            final_outcome = await self._final_processing(
                config.final_processing,
                initial_data,
                current_dialogue_result,  # Use cached conversation from step 2 - available throughout session
                fetched_data,
                None,  # No separate phase 2 result since it's continuous
                step3_response_body  # Pass response body separately
            )
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 4 OUTPUT: Final outcome length: {len(final_outcome) if final_outcome else 0} characters")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Final outcome preview: {final_outcome[:500] if final_outcome else 'None'}...")
            print(f"[SPECIAL FLOW SERVICE] Step 4 OUTPUT - Final outcome length: {len(final_outcome) if final_outcome else 0} chars, "
                  f"preview: {final_outcome[:200] if final_outcome else 'None'}...")

            # Step 5: Final API call
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== STEP 5: Final API call ==========")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Request tool ID: {config.final_api_call.request_tool_id}")
            print(f"[SPECIAL FLOW SERVICE] ========== STEP 5: Final API call ==========")
            print(f"[SPECIAL FLOW SERVICE] Request tool ID: {config.final_api_call.request_tool_id}")
            api_call_result = await self._final_api_call(
                config.final_api_call,
                final_outcome
            )
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 5 OUTPUT: API call result: {json.dumps(api_call_result, indent=2) if api_call_result else 'None'}")
            print(f"[SPECIAL FLOW SERVICE] Step 5 OUTPUT - API call success: {api_call_result.get('success') if isinstance(api_call_result, dict) else 'N/A'}, "
                  f"status: {api_call_result.get('status_code') if isinstance(api_call_result, dict) else 'N/A'}")

            total_time = time.time() - start_time
            print(f"[SPECIAL FLOW SERVICE] ========== FLOW EXECUTION COMPLETE ==========")
            print(f"[SPECIAL FLOW SERVICE] Total time: {total_time:.2f}s")
            return SpecialFlow1ExecuteResponse(
                flow_id=flow_id,
                flow_name=flow.name,
                success=True,
                phase="complete",
                initial_data=initial_data,
                dialogue_phase1=current_dialogue_result,  # Cached conversation from step 2
                fetched_data=fetched_data,
                dialogue_phase2=None,  # No dialogue phase 2 - only step 2 dialogue is used
                final_outcome=final_outcome,
                api_call_result=api_call_result,
                total_execution_time=total_time,
                error=None,
                metadata={},
            )

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Error executing dialogue-driven flow {flow_id}: {error_msg}", exc_info=True)
            print(f"[SPECIAL FLOW SERVICE] Error executing flow: {error_msg}")
            return SpecialFlow1ExecuteResponse(
                flow_id=flow_id,
                flow_name=flow.name,
                success=False,
                phase="error",
                initial_data=None,
                dialogue_phase1=None,
                fetched_data=None,
                dialogue_phase2=None,
                final_outcome=None,
                api_call_result=None,
                total_execution_time=total_time,
                error=error_msg,
                metadata={},
            )

    async def _start_dialogue_phase1(
        self,
        flow_id: str,
        config: "SpecialFlow1Config",
        initial_data: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Start dialogue phase 1 with initial data."""
        from .config import settings
        from .llm_factory import LLMFactory, LLMProvider
        from .llm_langchain_wrapper import LangChainLLMWrapper
        from .models import LLMProviderType

        if not self.dialogue_manager:
            raise ValueError("Dialogue manager not available")

        # Create a temporary dialogue profile for this flow
        dialogue_id = f"flow_{flow_id}_dialogue"
        
        # Build system prompt with initial data if needed
        system_prompt = config.dialogue_config.system_prompt
        if config.dialogue_config.use_initial_data and initial_data:
            initial_data_str = json.dumps(initial_data, indent=2) if isinstance(initial_data, dict) else str(initial_data)
            system_prompt = f"{system_prompt}\n\nInitial Data:\n{initial_data_str}"
            print(f"[SPECIAL FLOW SERVICE] Added initial data to system prompt: {len(initial_data_str)} chars")

        # Print the system prompt for step 2 (filter out formsample and pushsample, just show instructions)
        print(f"[SPECIAL FLOW SERVICE] ========== STEP 2 SYSTEM PROMPT ==========")
        # Filter out formsample and pushsample sections - extract just the instructions
        filtered_prompt = system_prompt
        
        # Remove formsample and pushsample sections using a more robust approach
        # This handles nested JSON objects by counting braces
        def remove_json_section(text, keyword):
            """Remove a JSON section starting with keyword"""
            pattern = rf'(?i){re.escape(keyword)}\s*:?\s*\{{'
            matches = list(re.finditer(pattern, text))
            for match in reversed(matches):  # Process from end to avoid index issues
                start = match.start()
                brace_count = 0
                i = match.end() - 1  # Start from the opening brace
                while i < len(text):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the matching closing brace
                            text = text[:start] + text[i+1:]
                            break
                    i += 1
            return text
        
        filtered_prompt = remove_json_section(filtered_prompt, 'formsample')
        filtered_prompt = remove_json_section(filtered_prompt, 'pushsample')
        
        # Clean up extra whitespace
        filtered_prompt = re.sub(r'\n\s*\n\s*\n+', '\n\n', filtered_prompt).strip()
        print(f"[SPECIAL FLOW SERVICE] {filtered_prompt}")
        print(f"[SPECIAL FLOW SERVICE] ===========================================")
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 2 SYSTEM PROMPT (filtered): {filtered_prompt}")

        # Determine provider/model
        provider = config.dialogue_config.llm_provider or LLMProviderType.GEMINI
        if provider == LLMProviderType.GEMINI:
            api_key = settings.gemini_api_key
            model_name = config.dialogue_config.model_name or settings.gemini_default_model
        elif provider == LLMProviderType.QWEN:
            api_key = settings.qwen_api_key
            model_name = config.dialogue_config.model_name or settings.qwen_default_model
        else:
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = settings.gemini_default_model

        print(f"[SPECIAL FLOW SERVICE] Using provider: {provider.value}, model: {model_name}")

        # Create or get dialogue profile for this flow
        from .models import DialogueProfile
        
        # Always recreate the profile to ensure system prompt changes take effect
        # This prevents caching issues where old system prompts might be used
        if dialogue_id in self.dialogue_manager.dialogues:
            # Clean up any active conversations for this profile to ensure fresh start
            conversations_to_remove = [
                conv_id for conv_id, conv in self.dialogue_manager.active_conversations.items()
                if conv.get("profile_id") == dialogue_id
            ]
            for conv_id in conversations_to_remove:
                del self.dialogue_manager.active_conversations[conv_id]
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Cleared active conversation: {conv_id}")
            # Remove old profile to force recreation with new system prompt
            del self.dialogue_manager.dialogues[dialogue_id]
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Removed existing dialogue profile to apply new system prompt")
            print(f"[SPECIAL FLOW SERVICE] Removed existing dialogue profile to apply new system prompt")
        
        # Create a new temporary dialogue profile with current config
        dialogue_profile = DialogueProfile(
            id=dialogue_id,
            name=f"Dialogue-Driven Flow Dialogue - {flow_id}",
            description=f"Temporary dialogue profile for Dialogue-Driven Flow: {flow_id}",
            system_prompt=system_prompt,
            rag_collection=None,  # DialogueConfig doesn't have rag_collection
            db_tools=[],  # DialogueConfig doesn't have db_tools
            request_tools=[],  # DialogueConfig doesn't have request_tools
            llm_provider=provider,
            model_name=model_name,
            max_turns=config.dialogue_config.max_turns_phase1,
            metadata={"flow_id": flow_id, "is_temporary": True}
        )
        self.dialogue_manager.dialogues[dialogue_id] = dialogue_profile
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Created/updated temporary dialogue profile: {dialogue_id} with system prompt length: {len(system_prompt)}")
        print(f"[SPECIAL FLOW SERVICE] Created/updated temporary dialogue profile: {dialogue_id} with system prompt length: {len(system_prompt)}")

        # Create LLM caller
        llm_caller = LLMFactory.create_caller(
            provider=LLMProvider(provider.value),
            api_key=api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=8192,
            timeout=settings.api_timeout,
        )
        llm = LangChainLLMWrapper(llm_caller=llm_caller)

        # Create conversation
        conversation_id = self.dialogue_manager._create_conversation(
            dialogue_id,
            "",  # No initial message - waiting for user
            turn_number=0
        )

        print(f"[SPECIAL FLOW SERVICE] Created conversation: {conversation_id}")

        # Return result indicating we need user input
        return {
            "conversation_id": conversation_id,
            "dialogue_id": dialogue_id,
            "system_prompt": system_prompt,
            "turn_number": 0,
            "max_turns": config.dialogue_config.max_turns_phase1,
            "response": "",
            "needs_more_info": True,
            "is_complete": False,
            "needs_user_input": True,
            "conversation_history": [],
            "llm_provider": provider.value,
            "model_name": model_name,
        }

    def _check_trigger_condition(
        self,
        trigger: "DataFetchTrigger",
        dialogue_result: Dict[str, Any],
    ) -> bool:
        """Check if trigger condition is met."""
        if trigger.type == "turn_count":
            turn_number = dialogue_result.get("turn_number", 0)
            return turn_number >= (trigger.value or 0)
        elif trigger.type == "keyword":
            # Check if keyword appears in conversation
            history = dialogue_result.get("conversation_history", [])
            keyword = str(trigger.value or "").lower()
            for msg in history:
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if keyword in content.lower():
                    return True
            return False
        elif trigger.type == "user_trigger":
            # User manually triggers - this would be handled by frontend
            return dialogue_result.get("user_triggered_fetch", False)
        elif trigger.type == "ai_detected":
            # AI detects need - check if response suggests need for data
            response = dialogue_result.get("response", "").lower()
            keywords = ["need", "require", "fetch", "get data", "retrieve"]
            return any(kw in response for kw in keywords)
        return False

    async def _fetch_mid_dialogue_data(
        self,
        request_config: "MidDialogueRequestConfig",
        dialogue_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch data using request tool based on dialogue context."""
        print(f"[SPECIAL FLOW SERVICE] Fetching mid-dialogue data from request tool: {request_config.request_tool_id}")
        
        if not self.request_tools_manager:
            raise ValueError("Request tools manager not available")

        profile = self.request_tools_manager.get_profile(request_config.request_tool_id)
        if not profile:
            raise ValueError(f"Request tool {request_config.request_tool_id} not found")

        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] _fetch_mid_dialogue_data: Extracting params from dialogue outcome")
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Using dialogue outcome (conversation_history) - this was defined by the dialogue prompt in step 2")
        
        conversation_history = dialogue_result.get("conversation_history", [])
        print(f"[SPECIAL FLOW SERVICE] Conversation history length: {len(conversation_history)}")

        # Map dialogue context to request params if param_mapping provided
        if request_config.param_mapping:
            # Get dialogue response for template replacement
            dialogue_response = dialogue_result.get("response", "")
            
            # Check if param_mapping is a string (direct template like "{{dialogue.response}}")
            if isinstance(request_config.param_mapping, str):
                # Direct template replacement
                template = request_config.param_mapping
                value = template.replace("{{dialogue.user_input}}", dialogue_result.get("user_input", ""))
                value = value.replace("{{dialogue.response}}", dialogue_response)
                if "{{dialogue.conversation_history}}" in value:
                    value = value.replace("{{dialogue.conversation_history}}", json.dumps(conversation_history))
                
                # Try to parse as JSON - if it's JSON, use it directly (don't merge with existing params)
                try:
                    parsed_json = json.loads(value)
                    if isinstance(parsed_json, dict):
                        # Use the parsed JSON directly as params (replace existing params)
                        # Exclude "value" key if it exists (redundant)
                        filtered_json = {k: v for k, v in parsed_json.items() if k != "value"}
                        profile.params = filtered_json
                        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Parsed JSON from template and set as params (excluding 'value'): {json.dumps(filtered_json, indent=2)}")
                        print(f"[SPECIAL FLOW SERVICE] Parsed JSON from template and set as params (excluding 'value'): {json.dumps(filtered_json, indent=2)[:200]}...")
                    else:
                        # Not a dict, use as single value
                        profile.params = parsed_json
                except (json.JSONDecodeError, ValueError):
                    # Not JSON, use as string value
                    # If template was just {{dialogue.response}}, use the response directly
                    if template.strip() == "{{dialogue.response}}":
                        # Try to parse dialogue_response as JSON
                        try:
                            parsed_response = json.loads(dialogue_response)
                            if isinstance(parsed_response, dict):
                                # Use the parsed JSON directly as params (replace existing params)
                                # Exclude "value" key if it exists (redundant)
                                filtered_response = {k: v for k, v in parsed_response.items() if k != "value"}
                                profile.params = filtered_response
                                self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Parsed dialogue.response as JSON and set as params (excluding 'value')")
                                print(f"[SPECIAL FLOW SERVICE] Parsed dialogue.response as JSON and set as params (excluding 'value')")
                            else:
                                profile.params = parsed_response
                        except (json.JSONDecodeError, ValueError):
                            # Not JSON, use as string in "value" key
                            profile.params = {"value": dialogue_response}
                    else:
                        # Other template, use the replaced value in "value" key
                        profile.params = {"value": value}
            else:
                # param_mapping is a dict - process each key-value pair
                # First, try to extract JSON from the conversation history
                # Look for JSON in assistant's last response
                extracted_json = None
                
                # Search for JSON in the conversation (usually in assistant's response)
                for msg in reversed(conversation_history):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # Try to find JSON in the content
                        json_match = re.search(r'\{[^{}]*"username"[^{}]*"formId"[^{}]*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                extracted_json = json.loads(json_match.group())
                                self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Extracted JSON from dialogue: {json.dumps(extracted_json, indent=2)}")
                                print(f"[SPECIAL FLOW SERVICE] Extracted JSON from dialogue: {json.dumps(extracted_json, indent=2)[:200]}...")
                                break
                            except:
                                pass
                        # Also try to find any JSON object
                        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                        for json_str in json_objects:
                            try:
                                parsed = json.loads(json_str)
                                if "username" in parsed and "formId" in parsed:
                                    extracted_json = parsed
                                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Extracted JSON from dialogue: {json.dumps(extracted_json, indent=2)}")
                                    print(f"[SPECIAL FLOW SERVICE] Extracted JSON from dialogue: {json.dumps(extracted_json, indent=2)[:200]}...")
                                    break
                            except:
                                pass
                        if extracted_json:
                            break
                
                # Apply param mapping
                for param_key, template in request_config.param_mapping.items():
                    value = template
                    
                    # If we extracted JSON, use it
                    if extracted_json and param_key in extracted_json:
                        value = extracted_json[param_key]
                        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Using extracted value for {param_key}: {value}")
                        print(f"[SPECIAL FLOW SERVICE] Using extracted value for {param_key}: {value}")
                    else:
                        # Try template replacement
                        value = value.replace("{{dialogue.user_input}}", dialogue_result.get("user_input", ""))
                        value = value.replace("{{dialogue.response}}", dialogue_response)
                        # Try to extract from conversation history
                        if "{{dialogue.conversation_history}}" in value:
                            value = value.replace("{{dialogue.conversation_history}}", json.dumps(conversation_history))
                        
                        # Special handling: if template was just {{dialogue.response}} and response is JSON, parse and merge
                        if template.strip() == "{{dialogue.response}}" and dialogue_response:
                            try:
                                parsed_response = json.loads(dialogue_response)
                                if isinstance(parsed_response, dict):
                                    # Merge the parsed JSON directly into params (don't use param_key)
                                    # Exclude "value" key if it exists (redundant)
                                    filtered_response = {k: v for k, v in parsed_response.items() if k != "value"}
                                    profile.params = {**(profile.params or {}), **filtered_response}
                                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Parsed dialogue.response as JSON and merged into params (skipping param_key, excluding 'value')")
                                    print(f"[SPECIAL FLOW SERVICE] Parsed dialogue.response as JSON and merged into params directly (excluding 'value')")
                                    continue  # Skip setting param_key since we merged directly
                                else:
                                    value = parsed_response
                            except (json.JSONDecodeError, ValueError):
                                # Not JSON, use as string
                                pass
                        else:
                            # Try to parse as JSON if it looks like JSON
                            try:
                                parsed_value = json.loads(value)
                                value = parsed_value
                            except:
                                pass
                    
                    profile.params[param_key] = value
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Set param {param_key} = {value}")
                    print(f"[SPECIAL FLOW SERVICE] Set param {param_key} = {value}")

        # Remove "value" key from params if it exists (redundant - params already contain all needed fields)
        if profile.params and "value" in profile.params:
            del profile.params["value"]
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Removed redundant 'value' key from params")
            print(f"[SPECIAL FLOW SERVICE] Removed redundant 'value' key from params")

        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Final request params: {json.dumps(profile.params, indent=2) if profile.params else 'None'}")
        print(f"[SPECIAL FLOW SERVICE] Final request params: {json.dumps(profile.params, indent=2) if profile.params else 'None'}")

        result = await asyncio.to_thread(
            self.request_tools_manager.execute_request,
            request_config.request_tool_id
        )
        print(f"[SPECIAL FLOW SERVICE] Request executed - success: {result.get('success') if isinstance(result, dict) else 'N/A'}")
        return result

    async def _continue_dialogue_with_data(
        self,
        flow_id: str,
        config: "SpecialFlow1Config",
        current_dialogue_result: Dict[str, Any],
        fetched_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Continue the same dialogue with fetched data injected."""
        from .config import settings
        from .llm_factory import LLMFactory, LLMProvider
        from .llm_langchain_wrapper import LangChainLLMWrapper
        from .models import LLMProviderType
        
        if not self.dialogue_manager:
            raise ValueError("Dialogue manager not available")
        
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] _continue_dialogue_with_data: Continuing dialogue with fetched data")
        print(f"[SPECIAL FLOW SERVICE] Continuing dialogue with fetched data")
        
        conversation_id = current_dialogue_result.get("conversation_id")
        dialogue_id = current_dialogue_result.get("dialogue_id")
        
        if not conversation_id or not dialogue_id:
            raise ValueError("Missing conversation_id or dialogue_id")
        
        # Get the conversation from dialogue manager
        conversation = self.dialogue_manager.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Get the dialogue profile
        profile = self.dialogue_manager.get_profile(dialogue_id)
        if not profile:
            raise ValueError(f"Dialogue profile {dialogue_id} not found")
        
        # Build context with fetched data
        system_prompt = profile.system_prompt
        if fetched_data:
            fetched_data_str = json.dumps(fetched_data, indent=2)
            system_prompt = f"{system_prompt}\n\nFetched Data:\n{fetched_data_str}"
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Injected fetched data into dialogue system prompt")
            print(f"[SPECIAL FLOW SERVICE] Injected fetched data into system prompt: {len(fetched_data_str)} chars")
        
        # Update profile temporarily
        original_system_prompt = profile.system_prompt
        profile.system_prompt = system_prompt
        # Use dialogue_phase2 config if available, otherwise use default
        max_turns = config.dialogue_phase2.max_turns_phase2 if config.dialogue_phase2 else 5
        profile.max_turns = max_turns
        
        # Determine provider/model
        provider = current_dialogue_result.get("llm_provider")
        model_name = current_dialogue_result.get("model_name")
        
        if provider:
            try:
                provider = LLMProviderType(provider)
            except:
                provider = LLMProviderType.GEMINI
        
        if provider == LLMProviderType.GEMINI:
            api_key = settings.gemini_api_key
            model_name = model_name or settings.gemini_default_model
        elif provider == LLMProviderType.QWEN:
            api_key = settings.qwen_api_key
            model_name = model_name or settings.qwen_default_model
        else:
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = settings.gemini_default_model
        
        # Create LLM caller
        llm_caller = LLMFactory.create_caller(
            provider=LLMProvider(provider.value),
            api_key=api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=8192,
            timeout=settings.api_timeout,
        )
        llm = LangChainLLMWrapper(llm_caller=llm_caller)
        
        # Get conversation history
        conversation_history = conversation.get("messages", [])
        current_turn = conversation.get("turn_number", 0)
        
        # Build conversation history string for prompt
        conversation_history_str = "\n".join([
            f"{msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')}: {msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')}"
            for msg in conversation_history
        ])
        
        # Build prompt with fetched data context
        if fetched_data:
            fetched_data_str = json.dumps(fetched_data, indent=2)
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation history so far:\n{conversation_history_str}\n\n"
                f"Now, based on the fetched data above, ask the user: 'What's the complaint against the user?' "
                f"or similar question to gather the complaint details."
            )
        else:
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation history so far:\n{conversation_history_str}\n\n"
                f"Now ask the user for the complaint details."
            )
        
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Calling LLM to continue dialogue with fetched data")
        print(f"[SPECIAL FLOW SERVICE] Calling LLM to continue dialogue")
        
        # Call LLM to generate the next message
        response_text = await llm.ainvoke(full_prompt)
        
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] AI response: {response_text[:200]}...")
        print(f"[SPECIAL FLOW SERVICE] LLM response: {response_text[:200]}...")
        
        # Add assistant response to conversation
        self.dialogue_manager._add_message_to_conversation(
            conversation_id,
            "assistant",
            response_text
        )
        
        # Increment turn
        self.dialogue_manager._increment_turn(conversation_id)
        
        # Get updated conversation
        updated_conversation = self.dialogue_manager.get_conversation(conversation_id)
        updated_history = updated_conversation.get("messages", [])
        
        # Determine if conversation needs to continue - improved logic to detect when conversation is actually complete
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
        
        max_turns = config.dialogue_phase2.max_turns_phase2 if config.dialogue_phase2 else 5
        
        # Determine if more info is needed
        # If AI has completion phrase and is not asking, it's complete
        # If AI is asking questions, it needs more info (unless we've hit max turns)
        needs_more_info = is_asking and not has_completion_phrase and updated_conversation["turn_number"] < max_turns
        is_complete = has_completion_phrase or (not needs_more_info) or updated_conversation["turn_number"] >= max_turns
        
        # Restore original system prompt
        profile.system_prompt = original_system_prompt
        
        return {
            "conversation_id": conversation_id,
            "dialogue_id": dialogue_id,
            "turn_number": updated_conversation.get("turn_number", current_turn + 1),
            "max_turns": config.dialogue_phase2.max_turns_phase2 if config.dialogue_phase2 else 5,
            "response": response_text,
            "needs_more_info": needs_more_info,
            "is_complete": is_complete,
            "needs_user_input": not is_complete,
            "conversation_history": [
                msg.model_dump() if hasattr(msg, 'model_dump') 
                else (msg if isinstance(msg, dict) 
                else {
                    "role": getattr(msg, 'role', 'unknown'), 
                    "content": getattr(msg, 'content', ''),
                    "timestamp": getattr(msg, 'timestamp', None)
                }) 
                for msg in updated_history
            ],
            "llm_provider": provider.value,
            "model_name": model_name,
        }

    async def _final_processing(
        self,
        processing_config: "FinalProcessingConfig",
        initial_data: Optional[Dict[str, Any]],
        dialogue_phase1_result: Optional[Dict[str, Any]],
        fetched_data: Optional[Dict[str, Any]],
        dialogue_phase2_result: Optional[Dict[str, Any]],
        step3_response_body: Optional[Any] = None,
    ) -> str:
        """Perform final processing with all data."""
        from .config import settings
        from .llm_factory import LLMFactory, LLMProvider
        from .llm_langchain_wrapper import LangChainLLMWrapper
        from .models import LLMProviderType

        print(f"[SPECIAL FLOW SERVICE] Final processing - initial_data: {bool(initial_data)}, "
              f"dialogue_phase1: {bool(dialogue_phase1_result)}, fetched_data: {bool(fetched_data)}, "
              f"step3_response_body: {bool(step3_response_body)}")

        # Extract response body from fetched_data if not provided separately
        if step3_response_body is None and isinstance(fetched_data, dict):
            step3_response_body = fetched_data.get('response_data') or fetched_data.get('response_body') or fetched_data.get('body') or fetched_data.get('data')
        
        # Print step 3 response body for debugging (short preview)
        if step3_response_body:
            body_preview = json.dumps(step3_response_body, indent=2) if isinstance(step3_response_body, dict) else str(step3_response_body)
            print(f"[SPECIAL FLOW SERVICE] Step 4 - Step 3 Response Body: {body_preview[:200]}..." if len(body_preview) > 200 else f"[SPECIAL FLOW SERVICE] Step 4 - Step 3 Response Body: {body_preview}")
        else:
            print(f"[SPECIAL FLOW SERVICE] Step 4 - Step 3 Response Body: None or not available")

        # Build input from template
        dialogue_summary = ""
        
        # Use dialogue_phase1_result which contains the cached conversation from step 2
        # This conversation is available throughout the session and was defined by the dialogue prompt
        if dialogue_phase1_result:
            history = dialogue_phase1_result.get("conversation_history", [])
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Using cached conversation from step 2: {len(history)} messages")
            print(f"[SPECIAL FLOW SERVICE] Using cached conversation: {len(history)} messages")
            dialogue_summary = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in history
            ])

        # Format step 3 response body for inclusion
        step3_response_body_str = ""
        if step3_response_body:
            if isinstance(step3_response_body, dict):
                step3_response_body_str = json.dumps(step3_response_body, indent=2)
            else:
                step3_response_body_str = str(step3_response_body)

        input_text = processing_config.input_template
        input_text = input_text.replace("{{initial_data}}", json.dumps(initial_data, indent=2) if initial_data else "None")
        input_text = input_text.replace("{{dialogue_summary}}", dialogue_summary)
        input_text = input_text.replace("{{fetched_data}}", json.dumps(fetched_data, indent=2) if fetched_data else "None")
        # Add step3_response_body if template supports it
        if "{{step3_response_body}}" in input_text:
            input_text = input_text.replace("{{step3_response_body}}", step3_response_body_str if step3_response_body_str else "None")
        
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Final processing input text length: {len(input_text)}")
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Final processing input preview: {input_text[:1000]}...")
        print(f"[SPECIAL FLOW SERVICE] Final processing input length: {len(input_text)} chars")

        # Ensure step 3 response body is available in step 4 context
        # Check if it's already in the system prompt or input template
        system_prompt = processing_config.system_prompt
        step3_already_in_context = False
        
        if step3_response_body_str:
            # Check if step 3 response is already mentioned in system prompt
            if "step 3" in system_prompt.lower() or "response body" in system_prompt.lower() or "response_data" in system_prompt.lower():
                step3_already_in_context = True
                print(f"[SPECIAL FLOW SERVICE] Step 3 response body already mentioned in system prompt")
            # Check if it's in input template via {{fetched_data}} or {{step3_response_body}}
            elif "{{fetched_data}}" in input_text or "{{step3_response_body}}" in input_text:
                step3_already_in_context = True
                print(f"[SPECIAL FLOW SERVICE] Step 3 response body available via input template ({{fetched_data}} or {{step3_response_body}})")
            
            # If not already in context, add it to system prompt
            if not step3_already_in_context:
                system_prompt = f"""{system_prompt}

Step 3 Response Data (from the request executed after dialogue):
{step3_response_body_str}"""
                print(f"[SPECIAL FLOW SERVICE] Added Step 3 response body to system prompt")

        # Determine provider/model
        provider = processing_config.llm_provider or LLMProviderType.GEMINI
        if provider == LLMProviderType.GEMINI:
            api_key = settings.gemini_api_key
            model_name = processing_config.model_name or settings.gemini_default_model
        elif provider == LLMProviderType.QWEN:
            api_key = settings.qwen_api_key
            model_name = processing_config.model_name or settings.qwen_default_model
        else:
            provider = LLMProviderType.GEMINI
            api_key = settings.gemini_api_key
            model_name = settings.gemini_default_model

        print(f"[SPECIAL FLOW SERVICE] Using provider: {provider.value}, model: {model_name}")

        # Create LLM caller
        llm_caller = LLMFactory.create_caller(
            provider=LLMProvider(provider.value),
            api_key=api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=8192,
            timeout=settings.api_timeout,
        )
        llm = LangChainLLMWrapper(llm_caller=llm_caller)

        # Build full prompt with explicit JSON output instruction
        full_prompt = f"""{system_prompt}

IMPORTANT: You must output ONLY valid JSON. Do not include any explanations, thinking, or markdown formatting. Output the JSON directly.

{input_text}

Remember: Output ONLY the JSON object, nothing else."""
        
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Final processing full prompt length: {len(full_prompt)}")
        print(f"[SPECIAL FLOW SERVICE] Calling LLM for final processing")

        # Call LLM
        response = await llm.ainvoke(full_prompt)
        
        print(f"[SPECIAL FLOW SERVICE] LLM response length: {len(response)} chars, preview: {response[:200]}...")
        
        # Try to extract JSON from response if it's wrapped in markdown or has extra text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                extracted_json = json.loads(json_match.group())
                response = json.dumps(extracted_json, indent=2)
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Extracted JSON from response")
                print(f"[SPECIAL FLOW SERVICE] Extracted JSON from response")
            except:
                self.logger.warning(f"[DIALOGUE-DRIVEN FLOW] Failed to extract JSON, using raw response")
                print(f"[SPECIAL FLOW SERVICE] Failed to extract JSON, using raw response")
        
        return response

    async def _final_api_call(
        self,
        api_config: "FinalAPICallConfig",
        final_outcome: str,
    ) -> Dict[str, Any]:
        """Make final API call with outcome."""
        print(f"[SPECIAL FLOW SERVICE] Making final API call with outcome length: {len(final_outcome)} chars")
        
        if not self.request_tools_manager:
            raise ValueError("Request tools manager not available")

        profile = self.request_tools_manager.get_profile(api_config.request_tool_id)
        if not profile:
            raise ValueError(f"Request tool {api_config.request_tool_id} not found")

        # Map final outcome to request body
        body_mapping = api_config.body_mapping.replace("{{final_outcome}}", final_outcome)
        try:
            profile.body = json.loads(body_mapping)
            print(f"[SPECIAL FLOW SERVICE] Parsed body as JSON")
        except:
            profile.body = body_mapping
            print(f"[SPECIAL FLOW SERVICE] Using body as string")
        
        # Print the request body for step 5
        print(f"[SPECIAL FLOW SERVICE] ========== STEP 5 REQUEST BODY ==========")
        if isinstance(profile.body, dict):
            body_str = json.dumps(profile.body, indent=2)
            print(f"[SPECIAL FLOW SERVICE] Request body (JSON): {body_str}")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Step 5 REQUEST BODY (JSON): {body_str}")
        else:
            body_str = str(profile.body)[:500]
            print(f"[SPECIAL FLOW SERVICE] Request body (string): {body_str}...")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Step 5 REQUEST BODY (string): {body_str}...")
        print(f"[SPECIAL FLOW SERVICE] =========================================")

        result = await asyncio.to_thread(
            self.request_tools_manager.execute_request,
            api_config.request_tool_id
        )
        print(f"[SPECIAL FLOW SERVICE] Final API call result - success: {result.get('success') if isinstance(result, dict) else 'N/A'}")
        return result

