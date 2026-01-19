"""
Flow Service - Main orchestration for workflow flows
"""
import logging
import os
import time
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
from .flow_step_executors import FlowStepExecutors


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

        # Initialize step executors
        self.step_executors = FlowStepExecutors(
            customization_manager=customization_manager,
            agent_manager=agent_manager,
            db_tools_manager=db_tools_manager,
            request_tools_manager=request_tools_manager,
            crawler_service=crawler_service,
            rag_system=rag_system,
            dialogue_manager=dialogue_manager,
        )

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
            print(f"[FLOW SERVICE] Loaded {len(self.flows)} flow profiles")
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
            print(f"[FLOW SERVICE] Saved {len(self.flows)} flow profiles")
        except Exception as e:
            self.logger.error(f"Error saving flows: {e}")

    def _format_flow_context(self, flow_context: List[Dict[str, Any]]) -> str:
        """
        Format accumulated flow context for inclusion in system prompts.
        
        Args:
            flow_context: List of context entries from previous steps
            
        Returns:
            Formatted context string with instructional header
        """
        if not flow_context:
            print(f"[FLOW CONTEXT] ⚠️  No flow context to format (empty)")
            return ""
        
        print(f"[FLOW CONTEXT] 🔨 Formatting flow context with {len(flow_context)} entries")
        
        context_parts = []
        context_parts.append("=" * 80)
        context_parts.append("FLOW CONTEXT - TASK HISTORY SO FAR")
        context_parts.append("=" * 80)
        context_parts.append("")
        context_parts.append("The following is the complete context of what has happened in this flow so far.")
        context_parts.append("This includes all conversations, AI responses, and key outputs from previous steps.")
        context_parts.append("You should use this context to understand the full picture of the task being performed.")
        context_parts.append("Even if some parts may not seem directly relevant to your current step, this context")
        context_parts.append("helps you understand the overall flow and make better decisions.")
        context_parts.append("")
        context_parts.append("-" * 80)
        context_parts.append("")
        
        for idx, entry in enumerate(flow_context, 1):
            step_name = entry.get("step_name", f"Step {idx}")
            step_type = entry.get("step_type", "unknown")
            context_parts.append(f"[{idx}] {step_name} ({step_type})")
            context_parts.append("-" * 80)
            
            # Format based on step type
            if step_type == "dialogue":
                # Format dialogue conversation
                conv_history = entry.get("conversation_history", [])
                if conv_history:
                    context_parts.append("Conversation:")
                    for msg in conv_history:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        if role == "user":
                            context_parts.append(f"  User: {content}")
                        elif role == "assistant":
                            context_parts.append(f"  Assistant: {content}")
                    context_parts.append("")
                response = entry.get("response", "")
                if response:
                    context_parts.append(f"Final Response: {response}")
                    context_parts.append("")
            
            elif step_type in ["customization", "agent"]:
                # Format AI response
                response = entry.get("response", entry.get("output", ""))
                if response:
                    context_parts.append(f"AI Response: {response}")
                    context_parts.append("")
            
            elif step_type in ["db_tool", "request", "crawler"]:
                # Format tool output summary
                output = entry.get("output", {})
                if isinstance(output, dict):
                    # Create a summary
                    summary_keys = list(output.keys())[:5]  # First 5 keys
                    context_parts.append(f"Output Summary: {', '.join(summary_keys)}")
                    if "success" in output:
                        context_parts.append(f"Success: {output.get('success')}")
                else:
                    output_str = str(output)[:200]
                    context_parts.append(f"Output: {output_str}...")
                context_parts.append("")
            
            context_parts.append("")
        
        context_parts.append("=" * 80)
        context_parts.append("END OF FLOW CONTEXT")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        formatted_context = "\n".join(context_parts)
        print(f"[FLOW CONTEXT] ✅ Formatted flow context: {len(formatted_context)} characters")
        print(f"[FLOW CONTEXT] 📊 Context breakdown:")
        for idx, entry in enumerate(flow_context, 1):
            step_type = entry.get("step_type", "unknown")
            step_name = entry.get("step_name", f"Step {idx}")
            print(f"[FLOW CONTEXT]   {idx}. {step_name} ({step_type})")
        
        return formatted_context
    
    def _add_to_flow_context(
        self,
        flow_context: List[Dict[str, Any]],
        step: FlowStepConfig,
        output: Union[str, Dict[str, Any]],
    ) -> None:
        """
        Add step output to flow context if it involves AI or dialogue.
        
        Args:
            flow_context: List to append to
            step: The step that was executed
            output: The output from the step
        """
        # Only add context for AI-involved steps
        if step.step_type not in [FlowStepType.CUSTOMIZATION, FlowStepType.AGENT, FlowStepType.DIALOGUE]:
            return
        
        context_entry = {
            "step_id": step.step_id,
            "step_name": step.step_name or step.step_id,
            "step_type": step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type),
        }
        
        if step.step_type == FlowStepType.DIALOGUE and isinstance(output, dict):
            # Extract dialogue information
            context_entry["conversation_history"] = output.get("conversation_history", [])
            context_entry["response"] = output.get("response", "")
            context_entry["is_complete"] = output.get("is_complete", False)
        elif step.step_type in [FlowStepType.CUSTOMIZATION, FlowStepType.AGENT]:
            # Extract AI response
            if isinstance(output, str):
                context_entry["response"] = output
            elif isinstance(output, dict):
                context_entry["response"] = output.get("response", output.get("output", str(output)))
            else:
                context_entry["response"] = str(output)
        else:
            # For other types, just store the output
            context_entry["output"] = output
        
        flow_context.append(context_entry)
        self.logger.info(
            f"[FLOW CONTEXT] Added {step.step_type.value} step '{step.step_id}' to flow context"
        )
        print(f"[FLOW CONTEXT] ✅ Added {step.step_type.value} step '{step.step_id}' ({step.step_name or step.step_id}) to flow context")
        print(f"[FLOW CONTEXT] 📊 Flow context now has {len(flow_context)} entries")
        
        # Print summary of what was added
        if step.step_type == FlowStepType.DIALOGUE and isinstance(output, dict):
            conv_history = context_entry.get("conversation_history", [])
            print(f"[FLOW CONTEXT]   - Dialogue conversation with {len(conv_history)} messages")
            response = context_entry.get("response", "")
            if response:
                print(f"[FLOW CONTEXT]   - Final response: {response[:100]}...")
        elif step.step_type in [FlowStepType.CUSTOMIZATION, FlowStepType.AGENT]:
            response = context_entry.get("response", "")
            if response:
                print(f"[FLOW CONTEXT]   - AI response: {response[:100]}...")

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
        print(f"[FLOW SERVICE] Listing {len(self.flows)} flows")
        return list(self.flows.values())

    def get_flow(self, flow_id: str) -> Optional[FlowProfile]:
        """Get a flow profile by ID."""
        result = self.flows.get(flow_id)
        print(f"[FLOW SERVICE] Get flow {flow_id}: {'found' if result else 'not found'}")
        return result

    def create_flow(self, req: FlowCreateRequest) -> str:
        """Create a new flow profile."""
        print(f"[FLOW SERVICE] Creating flow: {req.name}")
        print(f"[FLOW SERVICE] Flow steps: {len(req.steps) if req.steps else 0}")
        
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
        print(f"[FLOW SERVICE] Created flow profile: {flow_id}")
        return flow_id

    def update_flow(self, flow_id: str, req: FlowUpdateRequest) -> bool:
        """Update an existing flow profile."""
        print(f"[FLOW SERVICE] Updating flow: {flow_id}")
        
        if flow_id not in self.flows:
            print(f"[FLOW SERVICE] Flow {flow_id} not found")
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
            print(f"[FLOW SERVICE] Updated flow profile: {flow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating flow {flow_id}: {e}")
            print(f"[FLOW SERVICE] Error updating flow {flow_id}: {e}")
            return False

    def delete_flow(self, flow_id: str) -> bool:
        """Delete a flow profile."""
        print(f"[FLOW SERVICE] Deleting flow: {flow_id}")
        
        if flow_id not in self.flows:
            print(f"[FLOW SERVICE] Flow {flow_id} not found")
            return False
        try:
            del self.flows[flow_id]
            self.db.remove(self.query.id == flow_id)
            self.logger.info(f"Deleted flow profile: {flow_id}")
            print(f"[FLOW SERVICE] Deleted flow profile: {flow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting flow {flow_id}: {e}")
            print(f"[FLOW SERVICE] Error deleting flow {flow_id}: {e}")
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
        print(f"[FLOW SERVICE] ========== EXECUTING FLOW: {flow_id} ==========")
        print(f"[FLOW SERVICE] Initial input: {str(request.initial_input)[:200] if request.initial_input else 'None'}...")
        print(f"[FLOW SERVICE] Resume from step: {request.resume_from_step if request.resume_from_step else 'None'}")
        print(f"[FLOW SERVICE] Previous step results: {len(request.previous_step_results) if request.previous_step_results else 0}")
        
        if flow_id not in self.flows:
            raise ValueError(f"Flow {flow_id} not found")

        flow = self.flows[flow_id]
        if not flow.is_active:
            raise ValueError(f"Flow {flow_id} is not active")

        start_time = time.time()
        step_results: List[FlowStepResult] = request.previous_step_results.copy() if request.previous_step_results else []
        previous_output: Optional[Union[str, Dict[str, Any]]] = None
        
        # Initialize flow context accumulator for AI-involved steps
        flow_context: List[Dict[str, Any]] = []
        print(f"[FLOW CONTEXT] Initialized flow context accumulator for flow: {flow_id}")
        
        # If resuming, rebuild flow context from previous step results
        if request.previous_step_results:
            print(f"[FLOW CONTEXT] Rebuilding flow context from {len(request.previous_step_results)} previous step results")
            for result in request.previous_step_results:
                # Find the corresponding step
                step = next((s for s in flow.steps if s.step_id == result.step_id), None)
                if step and result.success and result.output:
                    self._add_to_flow_context(flow_context, step, result.output)
            print(f"[FLOW CONTEXT] Rebuilt flow context with {len(flow_context)} entries")

        # Log step_results when resuming
        if request.resume_from_step:
            self.logger.info(
                f"[FLOW {flow_id}] Resuming from step {request.resume_from_step}. "
                f"step_results count: {len(step_results)}, "
                f"step_ids: {[r.step_id for r in step_results]}"
            )
            print(f"[FLOW SERVICE] Resuming from step {request.resume_from_step}, "
                  f"previous results: {len(step_results)}")
        
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
                    print(f"[FLOW SERVICE] Re-executing dialogue step {resume_step.step_id}")
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
                        print(f"[FLOW SERVICE] Conversation ID: {conversation_id}")
                        
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
                                print(f"[FLOW SERVICE] Retrieved dialogue output from conversation, response: {last_response[:200]}...")
                            else:
                                # Conversation not found, try to re-execute
                                dialogue_output = await self.step_executors.execute_dialogue_step(
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
                                dialogue_output = await self.step_executors.execute_dialogue_step(
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
                                dialogue_output = await self.step_executors.execute_dialogue_step(
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
                        print(f"[FLOW SERVICE] Dialogue step enhanced - response: {response_text[:200]}..., "
                              f"user messages: {len(user_messages)}")
                    except Exception as e:
                        self.logger.warning(
                            f"[FLOW {flow_id}] Error re-executing dialogue step: {e}. "
                            f"Using previous step result output."
                        )
                        print(f"[FLOW SERVICE] Error re-executing dialogue step: {e}")
                        # Fallback to using the previous step result
                        if step_results:
                            last_result = step_results[-1]
                            if last_result.success and last_result.output:
                                previous_output = last_result.output
                    except Exception as e:
                        self.logger.error(f"Error processing dialogue step on resume: {e}")
                        print(f"[FLOW SERVICE] Error processing dialogue step on resume: {e}")
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
            print(f"[FLOW SERVICE] Starting execution from step {start_index}")
            
            for step_index, step in enumerate(flow.steps, 1):
                # Skip steps before resume point
                if step_index < start_index:
                    continue
                step_start = time.time()
                self.logger.info(
                    f"[FLOW {flow_id}] Starting step {step_index}/{len(flow.steps)}: {step.step_id} ({step.step_type})"
                )
                print(f"[FLOW SERVICE] ========== STEP {step_index}/{len(flow.steps)}: {step.step_id} ({step.step_type}) ==========")

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

                    print(f"[FLOW SERVICE] Step {step_index} input type: {type(step_input)}, "
                          f"preview: {str(step_input)[:200] if step_input else 'None'}...")

                    # Execute step based on type - ensure sequential execution
                    self.logger.info(
                        f"[FLOW {flow_id}] Executing step {step_index}/{len(flow.steps)}: {step.step_id} - waiting for completion..."
                    )
                    
                    # Prepare context with flow context for AI steps
                    step_context = request.context.copy() if request.context else {}
                    step_context["flow_context"] = flow_context
                    flow_context_formatted = self._format_flow_context(flow_context)
                    step_context["flow_context_formatted"] = flow_context_formatted
                    
                    # Print flow context info for AI-involved steps
                    if step.step_type in [FlowStepType.CUSTOMIZATION, FlowStepType.AGENT, FlowStepType.DIALOGUE]:
                        print(f"[FLOW CONTEXT] 🔄 Step {step_index} ({step.step_id}) is AI-involved - flow context available")
                        print(f"[FLOW CONTEXT] 📝 Flow context has {len(flow_context)} entries")
                        if flow_context_formatted:
                            context_preview = flow_context_formatted[:300] + "..." if len(flow_context_formatted) > 300 else flow_context_formatted
                            print(f"[FLOW CONTEXT] 📄 Formatted context preview ({len(flow_context_formatted)} chars):\n{context_preview}")
                        else:
                            print(f"[FLOW CONTEXT] ℹ️  No flow context yet (this is the first AI step)")
                    
                    output = await self._execute_step(
                        step, step_input, step_context
                    )
                    
                    # Add step output to flow context if it's AI-involved
                    if output:
                        self._add_to_flow_context(flow_context, step, output)
                    # Ensure step is fully complete before proceeding
                    self.logger.info(
                        f"[FLOW {flow_id}] Step {step_index}/{len(flow.steps)}: {step.step_id} completed successfully"
                    )
                    print(f"[FLOW SERVICE] Step {step_index} completed, output type: {type(output)}, "
                          f"preview: {str(output)[:200] if output else 'None'}...")

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
                            print(f"[FLOW SERVICE] Dialogue step waiting for user input - pausing flow")
                        else:
                            self.logger.info(
                                f"[FLOW {flow_id}] Dialogue step {step.step_id} is complete or ready to proceed. "
                                f"is_complete={is_complete}, conversation_id={conversation_id}"
                            )
                            print(f"[FLOW SERVICE] Dialogue step complete - is_complete={is_complete}")
                    
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
                        print(f"[FLOW SERVICE] Dialogue output enhanced - response: {response_text[:200]}..., "
                              f"user messages: {len(user_messages)}")
                        
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
                        print(f"[FLOW SERVICE] Flow paused at step {step_index}, total time: {total_time:.2f}s")
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
                    print(f"[FLOW SERVICE] Error executing step {step_index}: {error_msg}")
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

            print(f"[FLOW SERVICE] ========== FLOW EXECUTION COMPLETE ==========")
            print(f"[FLOW SERVICE] Success: {success}, Total time: {total_time:.2f}s, Steps: {len(step_results)}")
            print(f"[FLOW SERVICE] Final output type: {type(final_output)}, "
                  f"preview: {str(final_output)[:200] if final_output else 'None'}...")

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
            print(f"[FLOW SERVICE] Error executing flow: {error_msg}")
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
            return await self.step_executors.execute_customization_step(step, step_input, context)

        elif step.step_type == FlowStepType.AGENT:
            return await self.step_executors.execute_agent_step(step, step_input, context)

        elif step.step_type == FlowStepType.DB_TOOL:
            return await self.step_executors.execute_db_tool_step(step, step_input)

        elif step.step_type == FlowStepType.REQUEST:
            return await self.step_executors.execute_request_step(step, step_input)

        elif step.step_type == FlowStepType.CRAWLER:
            return await self.step_executors.execute_crawler_step(step, step_input)

        elif step.step_type == FlowStepType.DIALOGUE:
            return await self.step_executors.execute_dialogue_step(step, step_input, context)

        else:
            raise ValueError(f"Unknown step type: {step.step_type}")

