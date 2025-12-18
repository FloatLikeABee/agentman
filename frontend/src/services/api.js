import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// System Status
export const getStatus = async () => {
  const response = await api.get('/status');
  return response.data;
};

// RAG Collections
export const getRAGCollections = async () => {
  const response = await api.get('/rag/collections');
  return response.data;
};

export const addRAGData = async (payload) => {
  const { collection_name, data_input } = payload;
  const response = await api.post(`/rag/collections/${collection_name}/data`, data_input);
  return response.data;
};

export const validateRAGData = async (data) => {
  const response = await api.post('/rag/validate', data);
  return response.data;
};

export const queryRAGCollection = async (collectionName, query, nResults = 5) => {
  const response = await api.post(`/rag/collections/${collectionName}/query`, {
    query,
    n_results: nResults,
  });
  return response.data;
};

export const deleteRAGCollection = async (collectionName) => {
  const response = await api.delete(`/rag/collections/${collectionName}`);
  return response.data;
};

// Agents
export const getAgents = async () => {
  const response = await api.get('/agents');
  return response.data;
};

export const createAgent = async (config) => {
  const response = await api.post('/agents', config);
  return response.data;
};

export const updateAgent = async (agentId, config) => {
  const response = await api.put(`/agents/${agentId}`, config);
  return response.data;
};

export const deleteAgent = async (agentId) => {
  const response = await api.delete(`/agents/${agentId}`);
  return response.data;
};

export const runAgent = async (agentId, query, context = null) => {
  const response = await api.post(`/agents/${agentId}/run`, {
    query,
    context,
  });
  return response.data;
};

// Tools
export const getTools = async () => {
  const response = await api.get('/tools');
  return response.data;
};

export const updateToolConfig = async (payload) => {
  const { tool_id, config } = payload;
  const response = await api.put(`/tools/${tool_id}`, config);
  return response.data;
};

// Models
export const getModels = async () => {
  const response = await api.get('/models');
  return response.data;
};

// MCP
export const startMCPServer = async () => {
  const response = await api.post('/mcp/start');
  return response.data;
};

// Customizations
export const getCustomizations = async () => {
  const response = await api.get('/customizations');
  return response.data;
};

export const createCustomization = async (payload) => {
  const response = await api.post('/customizations', payload);
  return response.data;
};

export const deleteCustomization = async (profileId) => {
  const response = await api.delete(`/customizations/${profileId}`);
  return response.data;
};

export const queryCustomization = async (profileId, payload) => {
  const response = await api.post(`/customizations/${profileId}/query`, payload);
  return response.data;
};

// Export all functions
export default {
  getStatus,
  getRAGCollections,
  addRAGData,
  validateRAGData,
  queryRAGCollection,
  deleteRAGCollection,
  getAgents,
  createAgent,
  updateAgent,
  deleteAgent,
  runAgent,
  getTools,
  updateToolConfig,
  getModels,
  startMCPServer,
  getCustomizations,
  createCustomization,
  deleteCustomization,
  queryCustomization,
}; 