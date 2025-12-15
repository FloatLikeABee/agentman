import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from .config import settings
from .agent_manager import AgentManager
from .rag_system import RAGSystem
from .tools import ToolManager


class MCPService:
    def __init__(self, agent_manager: AgentManager, rag_system: RAGSystem, tool_manager: ToolManager):
        self.logger = logging.getLogger(__name__)
        self.agent_manager = agent_manager
        self.rag_system = rag_system
        self.tool_manager = tool_manager
        self.clients: Dict[str, Any] = {}

    async def start_server(self, host: str = "0.0.0.0", port: int = 8196):
        """Start the MCP server"""
        try:
            server = await asyncio.start_server(
                self._handle_client,
                host,
                port
            )
            
            self.logger.info(f"MCP Server started on {host}:{port}")
            
            async with server:
                await server.serve_forever()
                
        except Exception as e:
            self.logger.error(f"Error starting MCP server: {e}")
            raise

    async def _handle_client(self, reader, writer):
        """Handle incoming MCP client connections"""
        client_id = f"{writer.get_extra_info('peername')}"
        self.clients[client_id] = {
            'reader': reader,
            'writer': writer,
            'connected': True
        }
        
        self.logger.info(f"Client connected: {client_id}")
        
        try:
            while True:
                # Read message length
                length_bytes = await reader.read(4)
                if not length_bytes:
                    break
                    
                message_length = int.from_bytes(length_bytes, 'big')
                
                # Read message
                message_bytes = await reader.read(message_length)
                if not message_bytes:
                    break
                    
                message = json.loads(message_bytes.decode('utf-8'))
                
                # Process message
                response = await self._process_message(message, client_id)
                
                # Send response
                response_bytes = json.dumps(response).encode('utf-8')
                response_length = len(response_bytes).to_bytes(4, 'big')
                
                writer.write(response_length + response_bytes)
                await writer.drain()
                
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
            writer.close()
            await writer.wait_closed()
            self.logger.info(f"Client disconnected: {client_id}")

    async def _process_message(self, message: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Process incoming MCP messages"""
        try:
            message_type = message.get('type')
            
            if message_type == 'initialize':
                return await self._handle_initialize(message, client_id)
            elif message_type == 'tools/list':
                return await self._handle_tools_list(message, client_id)
            elif message_type == 'tools/call':
                return await self._handle_tools_call(message, client_id)
            elif message_type == 'rag/query':
                return await self._handle_rag_query(message, client_id)
            elif message_type == 'agent/run':
                return await self._handle_agent_run(message, client_id)
            elif message_type == 'ping':
                return {'type': 'pong'}
            else:
                return {
                    'type': 'error',
                    'error': {
                        'code': 'unknown_message_type',
                        'message': f'Unknown message type: {message_type}'
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                'type': 'error',
                'error': {
                    'code': 'internal_error',
                    'message': str(e)
                }
            }

    async def _handle_initialize(self, message: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle client initialization"""
        try:
            # Check LLM provider availability
            available_providers = self.agent_manager.get_available_providers()
            llm_available = len(available_providers) > 0
            available_models = self.agent_manager.get_available_models()

            # Get system status
            rag_collections = [col['name'] for col in self.rag_system.list_collections()]
            active_agents = [agent['id'] for agent in self.agent_manager.list_agents()]
            active_tools = [tool['id'] for tool in self.tool_manager.list_tools()]

            return {
                'type': 'initialize',
                'protocolVersion': '2024-11-05',
                'capabilities': {
                    'rag': True,
                    'tools': True,
                    'agents': True
                },
                'serverInfo': {
                    'name': 'RAG MCP Server',
                    'version': '1.0.0'
                },
                'status': {
                    'llm_available': llm_available,
                    'available_providers': available_providers,
                    'available_models': available_models,
                    'rag_collections': rag_collections,
                    'active_agents': active_agents,
                    'active_tools': active_tools
                }
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'error': {
                    'code': 'initialization_error',
                    'message': str(e)
                }
            }

    async def _handle_tools_list(self, message: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle tools list request"""
        try:
            tools = self.tool_manager.list_tools()
            return {
                'type': 'tools/list',
                'tools': tools
            }
        except Exception as e:
            return {
                'type': 'error',
                'error': {
                    'code': 'tools_list_error',
                    'message': str(e)
                }
            }

    async def _handle_tools_call(self, message: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle tool execution request"""
        try:
            tool_id = message.get('name')
            arguments = message.get('arguments', {})
            
            tool = self.tool_manager.get_tool(tool_id)
            if not tool:
                return {
                    'type': 'error',
                    'error': {
                        'code': 'tool_not_found',
                        'message': f'Tool {tool_id} not found'
                    }
                }
            
            # Execute tool
            result = tool.func(str(arguments))
            
            return {
                'type': 'tools/call',
                'content': [
                    {
                        'type': 'text',
                        'text': result
                    }
                ]
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'error': {
                    'code': 'tool_execution_error',
                    'message': str(e)
                }
            }

    async def _handle_rag_query(self, message: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle RAG query request"""
        try:
            collection_name = message.get('collection')
            query = message.get('query')
            n_results = message.get('n_results', 5)
            
            if not collection_name or not query:
                return {
                    'type': 'error',
                    'error': {
                        'code': 'missing_parameters',
                        'message': 'collection and query are required'
                    }
                }
            
            results = self.rag_system.query_collection(collection_name, query, n_results)
            
            return {
                'type': 'rag/query',
                'results': results
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'error': {
                    'code': 'rag_query_error',
                    'message': str(e)
                }
            }

    async def _handle_agent_run(self, message: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle agent execution request"""
        try:
            agent_id = message.get('agent_id')
            query = message.get('query')
            context = message.get('context')
            
            if not agent_id or not query:
                return {
                    'type': 'error',
                    'error': {
                        'code': 'missing_parameters',
                        'message': 'agent_id and query are required'
                    }
                }
            
            result = await self.agent_manager.run_agent(agent_id, query, context)
            
            return {
                'type': 'agent/run',
                'result': result
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'error': {
                    'code': 'agent_run_error',
                    'message': str(e)
                }
            }

    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        
        for client_id, client_data in self.clients.items():
            try:
                if client_data['connected']:
                    message_bytes = json.dumps(message).encode('utf-8')
                    message_length = len(message_bytes).to_bytes(4, 'big')
                    
                    client_data['writer'].write(message_length + message_bytes)
                    await client_data['writer'].drain()
                else:
                    disconnected_clients.append(client_id)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.clients:
                del self.clients[client_id] 