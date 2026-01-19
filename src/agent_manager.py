import asyncio
from langchain.agents import initialize_agent, AgentExecutor, create_react_agent
from langchain.agents import AgentType as LangChainAgentType
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import requests
import logging
import json
import os
from typing import Dict, List, Any, Optional
from tinydb import TinyDB, Query
from .config import settings
from .models import AgentConfig, ToolType, AgentType, LLMProviderType
from .rag_system import RAGSystem
from .tools import ToolManager
from .llm_factory import LLMFactory, LLMProvider
from .llm_langchain_wrapper import LangChainLLMWrapper

# Import all callers to register them with the factory
import gemini_caller
import qwen_caller
import mistral_caller


class AgentManager:
    def __init__(self, rag_system: RAGSystem, tool_manager: ToolManager):
        self.logger = logging.getLogger(__name__)
        self.rag_system = rag_system
        self.tool_manager = tool_manager
        self.agents: Dict[str, Any] = {}
        self.agents_db_path = os.path.join(settings.data_directory, "agents.json")
        os.makedirs(settings.data_directory, exist_ok=True)
        self.agents_db = TinyDB(self.agents_db_path)
        self.agent_query = Query()
        self._migrate_from_json_if_needed()
        self._load_agents()

    def _migrate_from_json_if_needed(self):
        """Migrate from old JSON file format to TinyDB if needed"""
        old_json_file = os.path.join(settings.data_directory, "agents.json")
        backup_file = os.path.join(settings.data_directory, "agents.json.backup")

        # Check if old JSON file exists and TinyDB is empty
        if os.path.exists(old_json_file) and len(self.agents_db) == 0:
            try:
                self.logger.info("Migrating agents from JSON file to TinyDB...")
                with open(old_json_file, 'r') as f:
                    agents_data = json.load(f)

                # Convert to TinyDB format
                for agent_id, agent_config in agents_data.items():
                    agent_doc = {
                        'id': agent_id,
                        'config': agent_config
                    }
                    self.agents_db.insert(agent_doc)

                # Create backup and remove old file
                import shutil
                shutil.copy2(old_json_file, backup_file)
                os.remove(old_json_file)

                self.logger.info(f"Successfully migrated {len(agents_data)} agents to TinyDB. Backup created at {backup_file}")
            except Exception as e:
                self.logger.error(f"Error migrating from JSON to TinyDB: {e}")

    def _load_agents(self):
        """Load agents from TinyDB storage"""
        try:
            # Get all agents from TinyDB
            agents_data = self.agents_db.all()
            for agent_doc in agents_data:
                try:
                    agent_id = agent_doc.get('id')
                    agent_config = agent_doc.get('config', {})
                    # Recreate the agent from config
                    config = AgentConfig(**agent_config)
                    self.create_agent(config)
                except Exception as e:
                    self.logger.error(f"Failed to load agent {agent_id}: {e}")
            self.logger.info(f"Loaded {len(self.agents)} agents from TinyDB")
        except Exception as e:
            self.logger.error(f"Error loading agents from TinyDB: {e}")

    def _save_agents(self):
        """Save agents to TinyDB storage"""
        try:
            # Clear existing data
            self.agents_db.truncate()

            # Save all current agents
            for agent_id, agent_data in self.agents.items():
                # Only save the config, not the runtime agent objects
                agent_doc = {
                    'id': agent_id,
                    'config': agent_data['config'].model_dump()
                }
                self.agents_db.insert(agent_doc)

            self.logger.info(f"Saved {len(self.agents)} agents to TinyDB")
        except Exception as e:
            self.logger.error(f"Error saving agents to TinyDB: {e}")

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return LLMFactory.get_available_providers()

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models for all providers"""
        models = []

        # Gemini models
        models.append({
            'name': settings.gemini_default_model,
            'provider': 'gemini',
            'description': 'Google Gemini model'
        })

        # Qwen models
        models.append({
            'name': settings.qwen_default_model,
            'provider': 'qwen',
            'description': 'Alibaba Qwen model'
        })

        # Mistral models
        models.append({
            'name': settings.mistral_default_model,
            'provider': 'mistral',
            'description': 'Mistral AI model'
        })

        return models

    def _create_llm_caller_with_fallback(self, preferred_provider: LLMProviderType, model: str, temperature: float, max_tokens: int):
        """Create LLM caller with fallback: try preferred first, only fallback on critical errors"""
        # First, try the preferred provider
        try:
            if preferred_provider == LLMProviderType.GEMINI:
                api_key = settings.gemini_api_key
                # Use appropriate model for Gemini
                model_to_use = model if model.startswith('gemini') else settings.gemini_default_model
            elif preferred_provider == LLMProviderType.QWEN:
                api_key = settings.qwen_api_key
                # Use appropriate model for Qwen
                model_to_use = model if model.startswith('qwen') else settings.qwen_default_model
            elif preferred_provider == LLMProviderType.MISTRAL:
                api_key = settings.mistral_api_key
                # Use appropriate model for Mistral
                model_to_use = model if model.startswith('mistral') else settings.mistral_default_model
            else:
                raise ValueError(f"Unknown provider: {preferred_provider}")

            # Check if API key is available
            if not api_key or api_key.strip() == "":
                raise ValueError(f"API key for {preferred_provider.value} is not configured")

            llm_caller = LLMFactory.create_caller(
                provider=LLMProvider(preferred_provider.value),
                api_key=api_key,
                model=model_to_use,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.logger.info(f"Successfully created {preferred_provider.value} caller with model {model_to_use}")
            return llm_caller
        except ValueError as e:
            # Critical error: missing API key or invalid provider - allow fallback
            self.logger.warning(f"Critical error with preferred provider {preferred_provider.value}: {e}. Attempting fallback...")
        except Exception as e:
            # For other errors, log but still try fallback (might be temporary network issues, etc.)
            self.logger.warning(f"Error creating {preferred_provider.value} caller: {e}. Attempting fallback...")

        # Fallback: try other providers only if preferred failed
        fallback_providers = [LLMProviderType.GEMINI, LLMProviderType.QWEN, LLMProviderType.MISTRAL]
        # Remove the preferred provider from fallback list
        fallback_providers = [p for p in fallback_providers if p != preferred_provider]

        for provider in fallback_providers:
            try:
                if provider == LLMProviderType.GEMINI:
                    api_key = settings.gemini_api_key
                    model_to_use = model if model.startswith('gemini') else settings.gemini_default_model
                elif provider == LLMProviderType.QWEN:
                    api_key = settings.qwen_api_key
                    model_to_use = model if model.startswith('qwen') else settings.qwen_default_model
                elif provider == LLMProviderType.MISTRAL:
                    api_key = settings.mistral_api_key
                    model_to_use = model if model.startswith('mistral') else settings.mistral_default_model
                else:
                    continue

                # Check if API key is available
                if not api_key or api_key.strip() == "":
                    self.logger.warning(f"Fallback provider {provider.value} has no API key configured, skipping...")
                    continue

                llm_caller = LLMFactory.create_caller(
                    provider=LLMProvider(provider.value),
                    api_key=api_key,
                    model=model_to_use,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                self.logger.warning(f"Using fallback provider {provider.value} with model {model_to_use} instead of {preferred_provider.value}")
                return llm_caller
            except Exception as e:
                self.logger.warning(f"Failed to create fallback {provider.value} caller: {e}")
                continue

        raise RuntimeError(f"Failed to create LLM caller for preferred provider {preferred_provider.value} and all fallback providers")

    def create_agent(self, config: AgentConfig) -> str:
        """Create a new agent with the specified configuration"""
        try:
            self.logger.info(f"Creating agent: {config.name}")
            self.logger.info(f"LLM Provider: {config.llm_provider}")
            self.logger.info(f"Model: {config.model_name}")
            self.logger.info(f"Tools count: {len(config.tools)}")
            self.logger.info(f"RAG collections: {config.rag_collections}")

            # Try to create LLM caller with fallback
            llm_caller = self._create_llm_caller_with_fallback(
                config.llm_provider,
                config.model_name,
                config.temperature,
                config.max_tokens
            )

            # Wrap in LangChain-compatible wrapper
            llm = LangChainLLMWrapper(llm_caller=llm_caller)

            # Get provider information for logging
            provider_info = llm_caller.get_model_info()
            provider_name = provider_info.get('provider', 'Unknown')
            model_name = provider_info.get('model', 'Unknown')
            
            # Normalize provider names for comparison
            configured_provider_lower = config.llm_provider.value.lower()
            actual_provider_lower = provider_name.lower()
            
            # Check if we're using the correct provider
            # Provider names: "GeminiCaller" contains "gemini", "QwenCaller" contains "qwen", "MistralCaller" contains "mistral"
            provider_match = (
                (configured_provider_lower == 'gemini' and 'gemini' in actual_provider_lower) or
                (configured_provider_lower == 'qwen' and 'qwen' in actual_provider_lower) or
                (configured_provider_lower == 'mistral' and 'mistral' in actual_provider_lower)
            )
            
            if not provider_match:
                self.logger.error(
                    f"❌ CRITICAL: Provider mismatch! Agent '{config.name}' configured for '{config.llm_provider.value}' "
                    f"but actually using '{provider_name}'. This indicates a fallback occurred. "
                    f"Please check your API keys and configuration."
                )
                # Still continue, but log the error clearly
            else:
                self.logger.info(f"✅ Agent '{config.name}' successfully using configured provider: {provider_name} with model {model_name}")

            # Build tools list
            tools = []
            
            # Add RAG tools if specified
            if config.rag_collections:
                for collection_name in config.rag_collections:
                    rag_tool = Tool(
                        name=f"RAG_{collection_name}",
                        func=lambda q, cn=collection_name: self._rag_query(cn, q),
                        description=f"Search the {collection_name} knowledge base for relevant information"
                    )
                    tools.append(rag_tool)

            # Add enabled tools
            for tool_name in config.tools:
                tool = self.tool_manager.get_tool(tool_name)
                if tool:
                    tools.append(tool)
                else:
                    self.logger.warning(f"Tool '{tool_name}' not found in tool manager. Available tools: {list(self.tool_manager.tools.keys())}")
            
            # Log tools being added
            if tools:
                self.logger.info(f"Agent '{config.name}' will have {len(tools)} tools: {[t.name for t in tools]}")
            else:
                self.logger.warning(f"Agent '{config.name}' has NO tools available! Will use direct LLM calls instead of agent executor.")
            
            # Only create agent executor if we have tools
            agent = None
            if tools:
                # Create agent using create_react_agent with proper prompt
                from langchain.agents import AgentExecutor, create_react_agent
                from langchain.prompts import PromptTemplate
                
            # Create an enhanced ReAct prompt template that encourages tool usage
            system_instruction = config.system_prompt or "You are a helpful AI assistant."
            
            # Inject system_prompt_data if provided (replace {data} placeholder)
            if config.system_prompt_data:
                if "{data}" in system_instruction:
                    system_instruction = system_instruction.replace("{data}", config.system_prompt_data)
                else:
                    # Append data to system prompt if no placeholder
                    system_instruction = f"{system_instruction}\n\nAdditional Data:\n{config.system_prompt_data}"
            
            if tools:
                system_instruction += " IMPORTANT: You have access to tools that can help you. ALWAYS use the available tools when needed. Do NOT say you cannot do something if you have a tool that can do it. Only use tools that are actually available in the tools list below."
                
                # Build tool names list for the prompt
                tool_names_str = ", ".join([t.name for t in tools])
                
                # Create template with tool_names as a placeholder variable
                # This is required by create_react_agent
                react_template = system_instruction + """

You have access to the following tools:

{tools}

IMPORTANT INSTRUCTIONS:
- ALWAYS use the available tools when they can help answer the question
- Do NOT say you cannot do something if you have a tool that can do it
- ONLY use tools that are listed above - do NOT try to use tools that are not in the list
- Available tool names: {tool_names}
- Read tool descriptions carefully to understand what each tool can do
- If a tool is not available, explain that you don't have access to that specific tool

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do and which tool to use
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

                prompt = PromptTemplate(
                    input_variables=["tools", "input", "agent_scratchpad"],
                    template=react_template,
                    partial_variables={"tool_names": tool_names_str}
                )
                
                # Create the agent with the required prompt
                # create_react_agent expects tool_names to be provided
                agent_prompt = create_react_agent(llm, tools, prompt)
                
                # Wrap in AgentExecutor with proper configuration
                # Enable parsing error handling to allow agent to recover from malformed outputs
                def handle_parsing_error(error: Exception) -> str:
                    """Handle parsing errors gracefully"""
                    self.logger.warning(f"Parsing error occurred: {error}. Attempting to recover...")
                    return f"Error parsing output: {str(error)}. Please try again with a clearer response format."

                agent = AgentExecutor(
                    agent=agent_prompt,
                    tools=tools,
                    verbose=True,
                    max_iterations=20,
                    early_stopping_method='force',
                    return_intermediate_steps=False,
                    handle_parsing_errors=handle_parsing_error,
                )
            else:
                # No tools available - agent will be None, and we'll use direct LLM calls
                self.logger.info(f"Agent '{config.name}' created without agent executor (no tools). Will use direct LLM calls.")

            # Store agent with provider information
            agent_id = config.name.lower().replace(' ', '_')
            self.agents[agent_id] = {
                'id': agent_id,
                'config': config,
                'agent': agent,  # Will be None if no tools available
                'llm': llm,
                'llm_caller': llm_caller,  # Store caller for provider info
                'provider': provider_name,
                'model': model_name,
                'has_tools': bool(tools)  # Track if agent has tools
            }

            self.logger.info(f"Created agent: {agent_id}")
            self._save_agents()
            return agent_id

        except Exception as e:
            self.logger.error(f"Error creating agent {config.name}: {e}")
            raise

    def _rag_query(self, collection_name: str, query: str) -> str:
        """Helper function for RAG queries"""
        try:
            results = self.rag_system.query_collection(collection_name, query)
            if results:
                context = "\n\n".join([r['content'] for r in results[:3]])
                return f"Based on the knowledge base, here's what I found:\n\n{context}"
            else:
                return "I couldn't find relevant information in the knowledge base."
        except Exception as e:
            self.logger.error(f"RAG query error: {e}")
            return "Error searching the knowledge base."

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents"""
        return [
            {
                'id': agent_id,
                'name': agent_data['config'].name,
                'description': agent_data['config'].description,
                'agent_type': agent_data['config'].agent_type,
                'model_name': agent_data['config'].model_name,
                'is_active': agent_data['config'].is_active,
                'rag_collections': agent_data['config'].rag_collections,
                'tools': agent_data['config'].tools
            }
            for agent_id, agent_data in self.agents.items()
        ]

    def update_agent(self, agent_id: str, config: AgentConfig) -> bool:
        """Update an existing agent"""
        try:
            if agent_id in self.agents:
                # Remove old agent
                del self.agents[agent_id]
                
                # Create new agent with updated config
                new_agent_id = self.create_agent(config)
                return new_agent_id == agent_id
            return False
        except Exception as e:
            self.logger.error(f"Error updating agent {agent_id}: {e}")
            return False

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent"""
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                # Also remove from TinyDB
                self.agents_db.remove(self.agent_query.id == agent_id)
                self.logger.info(f"Deleted agent: {agent_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting agent {agent_id}: {e}")
            return False

    async def run_agent(self, agent_id: str, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run an agent with a query"""
        try:
            agent_data = self.get_agent(agent_id)
            if not agent_data:
                raise ValueError(f"Agent {agent_id} not found")

            # Get provider information for logging
            agent_config = agent_data['config']
            configured_provider = agent_config.llm_provider.value
            configured_model = agent_config.model_name
            
            # Get actual provider being used
            actual_provider = agent_data.get('provider', 'Unknown')
            actual_model = agent_data.get('model', 'Unknown')
            
            # If we have llm_caller, get fresh info
            if 'llm_caller' in agent_data:
                caller_info = agent_data['llm_caller'].get_model_info()
                actual_provider = caller_info.get('provider', actual_provider)
                actual_model = caller_info.get('model', actual_model)
            
            # Log which AI is being used
            self.logger.info("=" * 60)
            self.logger.info(f"🤖 Running Agent: {agent_id}")
            self.logger.info(f"📋 Configured Provider: {configured_provider} | Model: {configured_model}")
            self.logger.info(f"⚙️ Actual Provider: {actual_provider} | Model: {actual_model}")
            if configured_provider.lower() not in actual_provider.lower():
                self.logger.warning(f"⚠️ WARNING: Provider mismatch detected! Using {actual_provider} instead of {configured_provider}")
            self.logger.info(f"💬 Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            self.logger.info("=" * 60)

            print(f"🤖 Running Agent: {agent_id}")
            print(f"📋 Configured Provider: {configured_provider} | Model: {configured_model}")
            print(f"⚙️ Actual Provider: {actual_provider} | Model: {actual_model}")

            # Add context to query if provided
            flow_context_formatted = None
            if context:
                # Check for flow context (from flow execution)
                flow_context_formatted = context.get("flow_context_formatted")
                
                # Check if system_prompt_data is in context (from flow execution)
                system_prompt_data = context.get("system_prompt_data")
                if system_prompt_data:
                    # Update agent's system prompt with data if needed
                    agent_config = agent_data.get('config')
                    if agent_config and hasattr(agent_config, 'system_prompt'):
                        # Temporarily inject data into system prompt
                        system_prompt = agent_config.system_prompt or ""
                        if isinstance(system_prompt_data, dict):
                            data_str = str(system_prompt_data)
                        else:
                            data_str = str(system_prompt_data)
                        
                        if "{data}" in system_prompt:
                            system_prompt = system_prompt.replace("{data}", data_str)
                        else:
                            system_prompt = f"{system_prompt}\n\nAdditional Data:\n{data_str}"
                        # Note: This won't update the stored agent, but will affect this execution
                        # For permanent updates, the agent config should be updated
                
                # Build context string excluding special keys
                context_items = {k: v for k, v in context.items() 
                               if k not in ["system_prompt_data", "flow_context_formatted", "flow_context"]}
                context_str = "\n".join([f"{k}: {v}" for k, v in context_items.items()])
                full_query = f"Context: {context_str}\n\nQuery: {query}" if context_str else query
            else:
                full_query = query
            
            # Prepend flow context to query if available (for both agent and direct LLM calls)
            if flow_context_formatted:
                print(f"[AGENT MANAGER] 📋 Prepending flow context to agent query ({len(flow_context_formatted)} chars)")
                full_query = f"{flow_context_formatted}\n\n{full_query}"
                print(f"[AGENT MANAGER] ✅ Flow context prepended - full query length: {len(full_query)} chars")

            # Check if agent has tools and agent executor is available
            has_tools = agent_data.get('has_tools', False)
            agent = agent_data.get('agent')

            if has_tools and agent is not None:
                # Use agent executor for tool-enabled queries
                try:
                    print('full_query:::::')
                    print(full_query)
                    response = agent.invoke({"input": full_query})

                    # Extract the response from the agent output
                    if isinstance(response, dict) and 'output' in response:
                        response_text = response['output']
                    elif isinstance(response, str):
                        response_text = response
                    else:
                        response_text = str(response)
                except Exception as invoke_error:
                    self.logger.warning(f"Agent invoke failed: {invoke_error}")
                    # Try direct LLM call as fallback
                    llm = agent_data['llm']
                    response_text = await llm.ainvoke(full_query)
                    print('llm-response:::::')
                    print(response_text)
            else:
                # No tools or agent executor not available, use direct LLM call
                if not has_tools:
                    self.logger.info(f"Agent '{agent_id}' has no tools configured, using direct LLM call")
                llm = agent_data['llm']
                response_text = await llm.ainvoke(full_query)

            return {
                'response': response_text,
                'agent_id': agent_id,
                'query': query,
                'context': context
            }

        except Exception as e:
            self.logger.error(f"Error running agent {agent_id}: {e}")
            return {
                'response': f"Error: {str(e)}",
                'agent_id': agent_id,
                'query': query,
                'error': True
            }

    async def run_agent_stream(self, agent_id: str, query: str, context: Optional[Dict[str, Any]] = None):
        """Run an agent with streaming response"""
        try:
            agent_data = self.get_agent(agent_id)
            if not agent_data:
                yield f"Error: Agent {agent_id} not found\n"
                return

            # Get provider information for logging
            agent_config = agent_data['config']
            configured_provider = agent_config.llm_provider.value
            configured_model = agent_config.model_name
            
            # Get actual provider being used
            actual_provider = agent_data.get('provider', 'Unknown')
            actual_model = agent_data.get('model', 'Unknown')
            
            # If we have llm_caller, get fresh info
            if 'llm_caller' in agent_data:
                caller_info = agent_data['llm_caller'].get_model_info()
                actual_provider = caller_info.get('provider', actual_provider)
                actual_model = caller_info.get('model', actual_model)
            
            # Log which AI is being used
            self.logger.info("=" * 60)
            self.logger.info(f"🤖 Running Agent (Stream): {agent_id}")
            self.logger.info(f"📋 Configured Provider: {configured_provider} | Model: {configured_model}")
            self.logger.info(f"⚙️  Actual Provider: {actual_provider} | Model: {actual_model}")
            if configured_provider.lower() not in actual_provider.lower():
                self.logger.warning(f"⚠️  WARNING: Provider mismatch detected! Using {actual_provider} instead of {configured_provider}")
            self.logger.info(f"💬 Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            self.logger.info("=" * 60)

            # Add context to query if provided
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                full_query = f"Context: {context_str}\n\nQuery: {query}"
            else:
                full_query = query

            # Check if agent has tools
            has_tools = bool(agent_config.rag_collections or agent_config.tools)

            if has_tools:
                # For agents with tools, run the agent and stream the final response
                try:
                    # Show that agent is thinking/processing
                    yield "🤔 **Agent is processing your request...**\n\n"

                    # Run agent (this may take time due to tool calls)
                    result = await self.run_agent(agent_id, query, context)

                    # Extract the response
                    response_text = result.get('response', 'No response generated')

                    # Stream the response word by word for better UX
                    words = response_text.split()
                    for i, word in enumerate(words):
                        yield word + " "
                        # Small delay to create streaming effect
                        if i % 10 == 0:  # Every 10 words
                            await asyncio.sleep(0.01)

                    yield "\n\n✅ **Agent execution complete**\n"

                except Exception as e:
                    self.logger.error(f"Agent streaming error: {e}")
                    yield f"❌ **Error during agent execution:** {str(e)}\n"
            else:
                # No tools, use direct LLM streaming
                llm = agent_data['llm']
                async for chunk in self._stream_llm_response(llm, full_query):
                    yield chunk

        except Exception as e:
            self.logger.error(f"Error running agent {agent_id}: {e}")
            yield f"Error: {str(e)}\n"

    async def _stream_llm_response(self, llm_wrapper, prompt: str):
        """Stream response from LLM wrapper"""
        try:
            # Get the underlying caller that supports streaming
            caller = llm_wrapper.llm_caller

            # Use the stream method (assuming it's synchronous for now)
            # In a real async implementation, you'd make the callers async
            for chunk in caller.stream(prompt):
                yield chunk

        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}\n"