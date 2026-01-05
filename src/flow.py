"""
Flow Service - Orchestrates workflows combining Customization, Agents, DBTools, Requests, and Crawler
"""
import logging
import os
import time
import asyncio
import json
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

        # Log step_results when resuming
        if request.resume_from_step:
            self.logger.info(
                f"[FLOW {flow_id}] Resuming from step {request.resume_from_step}. "
                f"step_results count: {len(step_results)}, "
                f"step_ids: {[r.step_id for r in step_results]}"
            )
        
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
                        # Get conversation_id from context or from previous step result
                        conversation_id = None
                        if request.context:
                            conversation_id = request.context.get("conversation_id")
                        
                        # Fallback: try to get conversation_id from previous step result
                        if not conversation_id and step_results:
                            for result in step_results:
                                if result.step_id == resume_step.step_id:
                                    # Check if it's a dialogue result with conversation_id
                                    if result.metadata and isinstance(result.metadata, dict):
                                        dialogue_response = result.metadata.get("dialogue_response")
                                        if dialogue_response and isinstance(dialogue_response, dict):
                                            conversation_id = dialogue_response.get("conversation_id")
                                    # Also check the output directly
                                    if not conversation_id and result.output and isinstance(result.output, dict):
                                        conversation_id = result.output.get("conversation_id")
                                    break
                        
                        self.logger.info(
                            f"[FLOW {flow_id}] Resuming dialogue step {resume_step.step_id}. "
                            f"conversation_id from context: {request.context.get('conversation_id') if request.context else None}, "
                            f"conversation_id found: {conversation_id}"
                        )
                        
                        if conversation_id and self.dialogue_manager:
                            # Check if conversation exists and get its data directly
                            conversation = self.dialogue_manager.get_conversation(conversation_id)
                            if conversation:
                                # Build dialogue output from existing conversation
                                messages = conversation.get("messages", [])
                                # Get the last assistant response
                                last_response = ""
                                for msg in reversed(messages):
                                    if hasattr(msg, "role") and msg.role == "assistant":
                                        last_response = getattr(msg, "content", "")
                                        break
                                    elif isinstance(msg, dict) and msg.get("role") == "assistant":
                                        last_response = msg.get("content", "")
                                        break
                                
                                # Get profile info
                                profile = self.dialogue_manager.get_profile(resume_step.resource_id)
                                profile_name = profile.name if profile else "Unknown"
                                
                                dialogue_output = {
                                    "conversation_id": conversation_id,
                                    "turn_number": conversation.get("turn_number", 0),
                                    "max_turns": conversation.get("max_turns", 999),
                                    "response": last_response,
                                    "needs_more_info": False,
                                    "is_complete": True,
                                    "profile_id": resume_step.resource_id,
                                    "profile_name": profile_name,
                                    "conversation_history": messages,
                                    "metadata": {}
                                }
                            else:
                                # Conversation not found, try to re-execute
                                dialogue_output = await self._execute_dialogue_step(
                                    resume_step,
                                    "",  # Empty initial message since conversation already started
                                    request.context or {}
                                )
                        else:
                            # No conversation_id or dialogue_manager
                            dialogue_output = None
                            
                            if not conversation_id:
                                self.logger.warning(
                                    f"[FLOW {flow_id}] No conversation_id found for dialogue step {resume_step.step_id}. "
                                    f"Cannot retrieve final conversation result. "
                                    f"Context keys: {list(request.context.keys()) if request.context else 'None'}, "
                                    f"Step results: {[r.step_id for r in step_results]}"
                                )
                                # Try to use the output from the previous step result if available
                                if step_results:
                                    for result in step_results:
                                        if result.step_id == resume_step.step_id and result.output:
                                            dialogue_output = result.output
                                            self.logger.info(
                                                f"[FLOW {flow_id}] Using output from previous step result for dialogue step {resume_step.step_id}"
                                            )
                                            break
                                
                                # If still no output, create a minimal output
                                if not dialogue_output:
                                    self.logger.error(
                                        f"[FLOW {flow_id}] Cannot retrieve dialogue output for step {resume_step.step_id}. "
                                        f"Creating minimal output."
                                    )
                                    dialogue_output = {
                                        "conversation_id": None,
                                        "turn_number": 0,
                                        "max_turns": 0,
                                        "response": "",
                                        "needs_more_info": False,
                                        "is_complete": True,
                                        "profile_id": resume_step.resource_id,
                                        "profile_name": "Unknown",
                                        "conversation_history": [],
                                        "metadata": {}
                                    }
                            elif not self.dialogue_manager:
                                # dialogue_manager is None, try to re-execute
                                self.logger.warning(
                                    f"[FLOW {flow_id}] Dialogue manager not available. "
                                    f"Attempting to re-execute dialogue step {resume_step.step_id}."
                                )
                                dialogue_output = await self._execute_dialogue_step(
                                    resume_step,
                                    "",  # Empty initial message since conversation already started
                                    request.context or {}
                                )
                            else:
                                # Both conversation_id and dialogue_manager are None/not available
                                # Try to re-execute as last resort
                                self.logger.warning(
                                    f"[FLOW {flow_id}] Both conversation_id and dialogue_manager unavailable. "
                                    f"Attempting to re-execute dialogue step {resume_step.step_id}."
                                )
                                dialogue_output = await self._execute_dialogue_step(
                                    resume_step,
                                    "",  # Empty initial message since conversation already started
                                    request.context or {}
                                )
                        
                        # Enhance dialogue output for next steps (same as in main execution)
                        enhanced_output = dialogue_output.copy()
                        response_text = dialogue_output.get("response", "")
                        conversation_history = dialogue_output.get("conversation_history", [])
                        
                        # Ensure conversation_history is a list of dicts (not Pydantic models)
                        serialized_history = []
                        for msg in conversation_history:
                            if isinstance(msg, dict):
                                serialized_history.append(msg)
                            elif hasattr(msg, "model_dump"):
                                serialized_history.append(msg.model_dump())
                            elif hasattr(msg, "dict"):
                                serialized_history.append(msg.dict())
                            elif hasattr(msg, "role") and hasattr(msg, "content"):
                                serialized_history.append({
                                    "role": getattr(msg, "role", "unknown"),
                                    "content": getattr(msg, "content", ""),
                                    "timestamp": getattr(msg, "timestamp", None)
                                })
                            else:
                                # Fallback: try to convert to dict
                                serialized_history.append(str(msg))
                        
                        user_messages = []
                        for msg in serialized_history:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                user_messages.append(msg.get("content", ""))
                        
                        enhanced_output["query"] = response_text
                        enhanced_output["body"] = response_text
                        if user_messages:
                            enhanced_output["user_input"] = user_messages[-1]
                            enhanced_output["all_user_messages"] = user_messages
                        enhanced_output["conversation_history"] = serialized_history
                        
                        # Find and update the dialogue step result in step_results
                        dialogue_step_result = None
                        for result in step_results:
                            if result.step_id == resume_step.step_id:
                                dialogue_step_result = result
                                break
                        
                        if dialogue_step_result:
                            # Update the existing step result with enhanced output
                            dialogue_step_result.output = enhanced_output
                            dialogue_step_result.metadata = {
                                "dialogue_response": {
                                    "conversation_id": dialogue_output.get("conversation_id"),
                                    "conversation_history": serialized_history,  # Use serialized history
                                    "response": response_text,
                                    "is_complete": dialogue_output.get("is_complete", True),
                                    "needs_more_info": False,
                                    "turn_number": dialogue_output.get("turn_number"),
                                    "max_turns": dialogue_output.get("max_turns"),
                                    "profile_id": dialogue_output.get("profile_id"),
                                    "profile_name": dialogue_output.get("profile_name"),
                                    "metadata": dialogue_output.get("metadata", {}),
                                }
                            }
                            dialogue_step_result.success = True
                            self.logger.info(
                                f"[FLOW {flow_id}] Updated existing dialogue step result {resume_step.step_id} in step_results"
                            )
                        else:
                            # Create a new step result if not found
                            dialogue_step_result = FlowStepResult(
                                step_id=resume_step.step_id,
                                step_name=resume_step.step_name,
                                step_type=FlowStepType.DIALOGUE,
                                success=True,
                                output=enhanced_output,
                                error=None,
                                execution_time=0.0,  # We don't have the original execution time
                                metadata={
                                    "dialogue_response": {
                                        "conversation_id": dialogue_output.get("conversation_id"),
                                        "conversation_history": serialized_history,
                                        "response": response_text,
                                        "is_complete": dialogue_output.get("is_complete", True),
                                        "needs_more_info": False,
                                        "turn_number": dialogue_output.get("turn_number"),
                                        "max_turns": dialogue_output.get("max_turns"),
                                        "profile_id": dialogue_output.get("profile_id"),
                                        "profile_name": dialogue_output.get("profile_name"),
                                        "metadata": dialogue_output.get("metadata", {}),
                                    }
                                }
                            )
                            step_results.append(dialogue_step_result)
                            self.logger.info(
                                f"[FLOW {flow_id}] Created new dialogue step result {resume_step.step_id} and added to step_results"
                            )
                        
                        # Set previous_output to the enhanced dialogue result
                        previous_output = enhanced_output
                        self.logger.info(
                            f"[FLOW {flow_id}] Dialogue step {resume_step.step_id} final result obtained and enhanced. "
                            f"Response: {response_text[:100]}..., User messages: {len(user_messages)}, "
                            f"History length: {len(serialized_history)}"
                        )
                        self.logger.debug(
                            f"[FLOW {flow_id}] Enhanced output keys: {list(enhanced_output.keys())}, "
                            f"previous_output type: {type(previous_output)}"
                        )
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
                                # Log what we're passing to help debug
                                self.logger.info(
                                    f"[FLOW {flow_id}] Step {step_index} ({step.step_id}) using previous output. "
                                    f"Type: {type(previous_output)}, Keys: {list(previous_output.keys()) if isinstance(previous_output, dict) else 'N/A'}"
                                )
                        else:
                            # Previous output is a string
                            step_input = str(previous_output)
                            self.logger.info(
                                f"[FLOW {flow_id}] Step {step_index} ({step.step_id}) using previous output as string: {step_input[:100]}..."
                            )
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
                        
                        # Ensure conversation_history is serialized for metadata
                        conv_history = output.get("conversation_history", [])
                        serialized_conv_history = []
                        for msg in conv_history:
                            if isinstance(msg, dict):
                                serialized_conv_history.append(msg)
                            elif hasattr(msg, "model_dump"):
                                serialized_conv_history.append(msg.model_dump())
                            elif hasattr(msg, "dict"):
                                serialized_conv_history.append(msg.dict())
                            elif hasattr(msg, "role") and hasattr(msg, "content"):
                                serialized_conv_history.append({
                                    "role": getattr(msg, "role", "unknown"),
                                    "content": getattr(msg, "content", ""),
                                    "timestamp": getattr(msg, "timestamp", None)
                                })
                            else:
                                serialized_conv_history.append(str(msg))
                        
                        metadata = {
                            "dialogue_response": {
                                "conversation_id": conversation_id,
                                "conversation_history": serialized_conv_history,  # Use serialized history
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
                    
                    # For dialogue steps, enhance the output to make it more usable for next steps
                    if step.step_type == FlowStepType.DIALOGUE and isinstance(output, dict):
                        # Create an enhanced output that includes both the full dialogue data
                        # and extracted fields that next steps can easily use
                        enhanced_output = output.copy()
                        
                        # Extract the final response text
                        response_text = output.get("response", "")
                        
                        # Extract user messages from conversation history for query parameters
                        conversation_history = output.get("conversation_history", [])
                        
                        # Ensure conversation_history is a list of dicts (not Pydantic models)
                        serialized_history = []
                        for msg in conversation_history:
                            if isinstance(msg, dict):
                                serialized_history.append(msg)
                            elif hasattr(msg, "model_dump"):
                                serialized_history.append(msg.model_dump())
                            elif hasattr(msg, "dict"):
                                serialized_history.append(msg.dict())
                            elif hasattr(msg, "role") and hasattr(msg, "content"):
                                serialized_history.append({
                                    "role": getattr(msg, "role", "unknown"),
                                    "content": getattr(msg, "content", ""),
                                    "timestamp": getattr(msg, "timestamp", None)
                                })
                            else:
                                # Fallback: try to convert to dict
                                serialized_history.append(str(msg))
                        
                        user_messages = []
                        for msg in serialized_history:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                user_messages.append(msg.get("content", ""))
                        
                        # Add helper fields for next steps to use
                        # These fields make it easier for request steps to extract data
                        enhanced_output["query"] = response_text  # Use response as query by default
                        enhanced_output["body"] = response_text  # Also available as body
                        
                        # Add extracted user input (last user message or all user messages)
                        if user_messages:
                            enhanced_output["user_input"] = user_messages[-1]  # Last user message
                            enhanced_output["all_user_messages"] = user_messages  # All user messages
                        
                        # Keep conversation_history accessible (use serialized version)
                        enhanced_output["conversation_history"] = serialized_history
                        
                        # Log what we're passing to next step
                        self.logger.info(
                            f"[FLOW {flow_id}] Dialogue step {step.step_id} output enhanced for next step. "
                            f"Response: {response_text[:100]}..., User messages: {len(user_messages)}, "
                            f"History length: {len(serialized_history)}"
                        )
                        self.logger.debug(
                            f"[FLOW {flow_id}] Enhanced output for step {step.step_id}: "
                            f"keys={list(enhanced_output.keys())}, "
                            f"has_query={('query' in enhanced_output)}, "
                            f"has_user_input={('user_input' in enhanced_output)}"
                        )
                        
                        output = enhanced_output
                    
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
        self.flows: Dict[str, "SpecialFlow1Profile"] = {}
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
                    from .models import SpecialFlow1Profile, SpecialFlow1Config
                    profile = SpecialFlow1Profile(id=flow_id, **data)
                    self.flows[flow_id] = profile
                except Exception as e:
                    self.logger.error(f"Failed to load flow {doc.get('id')}: {e}")
            self.logger.info(f"Loaded {len(self.flows)} dialogue-driven flow profiles")
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

    def list_flows(self) -> List["SpecialFlow1Profile"]:
        """List all flow profiles."""
        return list(self.flows.values())

    def get_flow(self, flow_id: str) -> Optional["SpecialFlow1Profile"]:
        """Get a flow profile by ID."""
        return self.flows.get(flow_id)

    def create_flow(self, req: "SpecialFlow1CreateRequest") -> str:
        """Create a new flow profile."""
        from .models import SpecialFlow1Profile
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
        return flow_id

    def update_flow(self, flow_id: str, req: "SpecialFlow1UpdateRequest") -> bool:
        """Update an existing flow profile."""
        if flow_id not in self.flows:
            return False
        try:
            from .models import SpecialFlow1Profile
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
            self.logger.info(f"Deleted dialogue-driven flow profile: {flow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting flow {flow_id}: {e}")
            return False

    async def execute_flow(
        self, flow_id: str, request: "SpecialFlow1ExecuteRequest"
    ) -> "SpecialFlow1ExecuteResponse":
        """
        Execute a Dialogue-Driven Flow.
        
        Steps:
        1. Fetch initial data (DB tool or Request tool)
        2. Start dialogue with initial data (caches all conversation)
        3. Fetch data after dialogue (Request tool, uses cached conversation)
        4. Final processing with all data (uses cached conversation)
        5. Call final API
        """
        from .models import (
            SpecialFlow1ExecuteResponse,
            SpecialFlow1Profile,
        )
        
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
                # Cache the dialogue conversation for the entire session - step 3 and step 4 will use this cached conversation
                # The conversation_history contains the full dialogue outcome from step 2, which is defined by the dialogue prompt
                current_dialogue_result = dialogue_phase1_result
                conversation_id = current_dialogue_result.get("conversation_id")
                dialogue_id = current_dialogue_result.get("dialogue_id")
                
                # Log that conversation is cached for later steps
                if dialogue_phase1_result.get('conversation_history'):
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Using cached conversation from previous session - {len(dialogue_phase1_result.get('conversation_history', []))} messages available for steps 3 and 4")
            else:
                # Step 1: Fetch initial data
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== STEP 1: Fetching initial data ==========")
                initial_data = None
                if config.initial_data_source.type == "db_tool":
                    if not self.db_tools_manager:
                        raise ValueError("Database tools manager not available")
                    sql_input = request.initial_input if request.initial_input else config.initial_data_source.sql_input
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Using DB tool: {config.initial_data_source.resource_id}, SQL: {sql_input}")
                    initial_data = await asyncio.to_thread(
                        self.db_tools_manager.execute_query,
                        tool_id=config.initial_data_source.resource_id,
                        sql_input=sql_input,
                        force_refresh=True,
                    )
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 1 OUTPUT: Initial data fetched: {json.dumps(initial_data, indent=2) if initial_data else 'None'}")
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
                            except Exception as e:
                                self.logger.warning(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Failed to parse initial_input: {e}")
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Using Request tool: {config.initial_data_source.resource_id}")
                    initial_data = await asyncio.to_thread(
                        self.request_tools_manager.execute_request,
                        config.initial_data_source.resource_id
                    )
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 1 OUTPUT: Initial data fetched: {json.dumps(initial_data, indent=2) if initial_data else 'None'}")

                # Step 2: Start dialogue (continuous - stays open until step 7)
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== STEP 2: Starting dialogue ==========")
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] System prompt: {config.dialogue_config.system_prompt[:200]}...")
                dialogue_phase1_result = await self._start_dialogue_phase1(
                    flow_id, config, initial_data, request.context or {}
                )
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 2 OUTPUT: Dialogue result - conversation_id: {dialogue_phase1_result.get('conversation_id')}, needs_user_input: {dialogue_phase1_result.get('needs_user_input')}, is_complete: {dialogue_phase1_result.get('is_complete')}")
                if dialogue_phase1_result.get('conversation_history'):
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Conversation history length: {len(dialogue_phase1_result.get('conversation_history', []))}")

                # Cache dialogue conversation for the entire session - step 3 and step 4 will use this cached conversation
                # The conversation_history contains the full dialogue outcome from step 2, which is defined by the dialogue prompt
                current_dialogue_result = dialogue_phase1_result
                conversation_id = dialogue_phase1_result.get("conversation_id")
                dialogue_id = dialogue_phase1_result.get("dialogue_id")
                
                # Log that conversation is cached for later steps
                if dialogue_phase1_result.get('conversation_history'):
                    self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Cached conversation for session - {len(dialogue_phase1_result.get('conversation_history', []))} messages will be available for steps 3 and 4")
                
                # Check if we need to pause for initial user interaction
                if dialogue_phase1_result.get("needs_user_input") and not dialogue_phase1_result.get("is_complete"):
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
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Param mapping: {json.dumps(config.mid_dialogue_request.param_mapping, indent=2) if config.mid_dialogue_request.param_mapping else 'None'}")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Using dialogue outcome (conversation_history) from step 2 - defined by dialogue prompt")
            if current_dialogue_result.get('conversation_history'):
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Conversation history available: {len(current_dialogue_result.get('conversation_history', []))} messages")
            fetched_data = await self._fetch_mid_dialogue_data(
                config.mid_dialogue_request,
                current_dialogue_result  # Use cached conversation from step 2 - contains dialogue outcome defined by prompt
            )
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 3 OUTPUT: Fetched data: {json.dumps(fetched_data, indent=2) if fetched_data else 'None'}")

            # Step 4: Final processing (use cached conversation from step 2 - available throughout the session)
            # The full conversation history from step 2 is cached and used here
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== STEP 4: Final processing ==========")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] System prompt: {config.final_processing.system_prompt[:200]}...")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Input template: {config.final_processing.input_template[:200]}...")
            if current_dialogue_result.get('conversation_history'):
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Using cached conversation from step 2: {len(current_dialogue_result.get('conversation_history', []))} messages")
            final_outcome = await self._final_processing(
                config.final_processing,
                initial_data,
                current_dialogue_result,  # Use cached conversation from step 2 - available throughout session
                fetched_data,
                None  # No separate phase 2 result since it's continuous
            )
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 4 OUTPUT: Final outcome length: {len(final_outcome) if final_outcome else 0} characters")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Final outcome preview: {final_outcome[:500] if final_outcome else 'None'}...")

            # Step 5: Final API call
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] ========== STEP 5: Final API call ==========")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Request tool ID: {config.final_api_call.request_tool_id}")
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Body mapping: {config.final_api_call.body_mapping}")
            api_call_result = await self._final_api_call(
                config.final_api_call,
                final_outcome
            )
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Step 5 OUTPUT: API call result: {json.dumps(api_call_result, indent=2) if api_call_result else 'None'}")

            total_time = time.time() - start_time
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

        # Create or get dialogue profile for this flow
        from .models import DialogueProfile
        if dialogue_id not in self.dialogue_manager.dialogues:
            # Create a temporary dialogue profile
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
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] Created temporary dialogue profile: {dialogue_id}")
        else:
            # Update existing profile with current config
            existing_profile = self.dialogue_manager.dialogues[dialogue_id]
            existing_profile.system_prompt = system_prompt
            existing_profile.max_turns = config.dialogue_config.max_turns_phase1
            existing_profile.llm_provider = provider
            existing_profile.model_name = model_name
            # DialogueConfig doesn't have rag_collection, db_tools, or request_tools

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
        import re
        
        if not self.request_tools_manager:
            raise ValueError("Request tools manager not available")

        profile = self.request_tools_manager.get_profile(request_config.request_tool_id)
        if not profile:
            raise ValueError(f"Request tool {request_config.request_tool_id} not found")

        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] _fetch_mid_dialogue_data: Extracting params from dialogue outcome")
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Using dialogue outcome (conversation_history) - this was defined by the dialogue prompt in step 2")
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Dialogue conversation history: {json.dumps(dialogue_result.get('conversation_history', []), indent=2)}")

        # Map dialogue context to request params if param_mapping provided
        if request_config.param_mapping:
            # First, try to extract JSON from the conversation history
            # Look for JSON in assistant's last response
            conversation_history = dialogue_result.get("conversation_history", [])
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
                else:
                    # Try template replacement
                    value = value.replace("{{dialogue.user_input}}", dialogue_result.get("user_input", ""))
                    value = value.replace("{{dialogue.response}}", dialogue_result.get("response", ""))
                    # Try to extract from conversation history
                    if "{{dialogue.conversation_history}}" in value:
                        value = value.replace("{{dialogue.conversation_history}}", json.dumps(conversation_history))
                    # Try to parse as JSON if it looks like JSON
                    try:
                        parsed_value = json.loads(value)
                        value = parsed_value
                    except:
                        pass
                
                profile.params[param_key] = value
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Set param {param_key} = {value}")

        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Final request params: {json.dumps(profile.params, indent=2) if profile.params else 'None'}")

        result = await asyncio.to_thread(
            self.request_tools_manager.execute_request,
            request_config.request_tool_id
        )
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
        
        # Call LLM to generate the next message
        response_text = await llm.ainvoke(full_prompt)
        
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW {flow_id}] AI response: {response_text[:200]}...")
        
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
        
        # Determine if conversation needs to continue
        response_lower = response_text.lower()
        asking_phrases = ["?", "can you", "could you", "please provide", "i need", "what", "which", "when", "where", "how"]
        max_turns = config.dialogue_phase2.max_turns_phase2 if config.dialogue_phase2 else 5
        needs_more_info = any(phrase in response_lower for phrase in asking_phrases) and updated_conversation["turn_number"] < max_turns
        is_complete = not needs_more_info or updated_conversation["turn_number"] >= max_turns
        
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
    ) -> str:
        """Perform final processing with all data."""
        from .config import settings
        from .llm_factory import LLMFactory, LLMProvider
        from .llm_langchain_wrapper import LangChainLLMWrapper
        from .models import LLMProviderType

        # Build input from template
        dialogue_summary = ""
        
        # Use dialogue_phase1_result which contains the cached conversation from step 2
        # This conversation is available throughout the session and was defined by the dialogue prompt
        if dialogue_phase1_result:
            history = dialogue_phase1_result.get("conversation_history", [])
            self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Using cached conversation from step 2: {len(history)} messages")
            dialogue_summary = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in history
            ])

        input_text = processing_config.input_template
        input_text = input_text.replace("{{initial_data}}", json.dumps(initial_data, indent=2) if initial_data else "None")
        input_text = input_text.replace("{{dialogue_summary}}", dialogue_summary)
        input_text = input_text.replace("{{fetched_data}}", json.dumps(fetched_data, indent=2) if fetched_data else "None")
        
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Final processing input text length: {len(input_text)}")
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Final processing input preview: {input_text[:1000]}...")

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
        full_prompt = f"""{processing_config.system_prompt}

IMPORTANT: You must output ONLY valid JSON. Do not include any explanations, thinking, or markdown formatting. Output the JSON directly.

{input_text}

Remember: Output ONLY the JSON object, nothing else."""
        
        self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Final processing full prompt length: {len(full_prompt)}")

        # Call LLM
        response = await llm.ainvoke(full_prompt)
        
        # Try to extract JSON from response if it's wrapped in markdown or has extra text
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                extracted_json = json.loads(json_match.group())
                response = json.dumps(extracted_json, indent=2)
                self.logger.info(f"[DIALOGUE-DRIVEN FLOW] Extracted JSON from response")
            except:
                self.logger.warning(f"[DIALOGUE-DRIVEN FLOW] Failed to extract JSON, using raw response")
        
        return response

    async def _final_api_call(
        self,
        api_config: "FinalAPICallConfig",
        final_outcome: str,
    ) -> Dict[str, Any]:
        """Make final API call with outcome."""
        if not self.request_tools_manager:
            raise ValueError("Request tools manager not available")

        profile = self.request_tools_manager.get_profile(api_config.request_tool_id)
        if not profile:
            raise ValueError(f"Request tool {api_config.request_tool_id} not found")

        # Map final outcome to request body
        body_mapping = api_config.body_mapping.replace("{{final_outcome}}", final_outcome)
        try:
            profile.body = json.loads(body_mapping)
        except:
            profile.body = body_mapping

        result = await asyncio.to_thread(
            self.request_tools_manager.execute_request,
            api_config.request_tool_id
        )
        return result

