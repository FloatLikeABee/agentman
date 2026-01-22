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

export const getAgent = async (agentId) => {
  const response = await api.get(`/agents/${agentId}`);
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

export const getProviders = async () => {
  const response = await api.get('/providers');
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

export const updateCustomization = async (profileId, payload) => {
  const response = await api.put(`/customizations/${profileId}`, payload);
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

// Dialogues
export const getDialogues = async () => {
  const response = await api.get('/dialogues');
  return response.data;
};

export const getDialogue = async (dialogueId) => {
  const response = await api.get(`/dialogues/${dialogueId}`);
  return response.data;
};

export const createDialogue = async (payload) => {
  const response = await api.post('/dialogues', payload);
  return response.data;
};

export const updateDialogue = async (dialogueId, payload) => {
  const response = await api.put(`/dialogues/${dialogueId}`, payload);
  return response.data;
};

export const deleteDialogue = async (dialogueId) => {
  const response = await api.delete(`/dialogues/${dialogueId}`);
  return response.data;
};

export const startDialogue = async (dialogueId, payload) => {
  const response = await api.post(`/dialogues/${dialogueId}/start`, payload);
  return response.data;
};

export const continueDialogue = async (dialogueId, payload) => {
  const response = await api.post(`/dialogues/${dialogueId}/continue`, payload);
  return response.data;
};

// Crawler
export const crawlWebsite = async (payload) => {
  const response = await api.post('/crawler/crawl', payload);
  return response.data;
};

// Crawler Profiles
export const getCrawlerProfiles = async () => {
  const response = await api.get('/crawler/profiles');
  return response.data;
};

export const getCrawlerProfile = async (profileId) => {
  const response = await api.get(`/crawler/profiles/${profileId}`);
  return response.data;
};

export const createCrawlerProfile = async (payload) => {
  const response = await api.post('/crawler/profiles', payload);
  return response.data;
};

export const updateCrawlerProfile = async (profileId, payload) => {
  const response = await api.put(`/crawler/profiles/${profileId}`, payload);
  return response.data;
};

export const deleteCrawlerProfile = async (profileId) => {
  const response = await api.delete(`/crawler/profiles/${profileId}`);
  return response.data;
};

export const executeCrawlerProfile = async (profileId) => {
  const response = await api.post(`/crawler/profiles/${profileId}/execute`);
  return response.data;
};

// Database Tools
export const getDBTools = async () => {
  const response = await api.get('/db-tools');
  return response.data;
};

export const getDBTool = async (toolId) => {
  const response = await api.get(`/db-tools/${toolId}`);
  return response.data;
};

export const createDBTool = async (payload) => {
  const response = await api.post('/db-tools', payload);
  return response.data;
};

export const updateDBTool = async (toolId, payload) => {
  const response = await api.put(`/db-tools/${toolId}`, payload);
  return response.data;
};

export const deleteDBTool = async (toolId) => {
  const response = await api.delete(`/db-tools/${toolId}`);
  return response.data;
};

export const previewDBTool = async (toolId, forceRefresh = false) => {
  const response = await api.get(`/db-tools/${toolId}/preview`, {
    params: { force_refresh: forceRefresh }
  });
  return response.data;
};

export const executeDBTool = async (toolId, sqlInput = null) => {
  const response = await api.post(`/db-tools/${toolId}/execute`, {
    sql_input: sqlInput
  });
  return response.data;
};

// Request Tools
export const getRequestTools = async () => {
  const response = await api.get('/request-tools');
  return response.data;
};

export const getRequestTool = async (requestId) => {
  const response = await api.get(`/request-tools/${requestId}`);
  return response.data;
};

export const createRequestTool = async (payload) => {
  const response = await api.post('/request-tools', payload);
  return response.data;
};

export const updateRequestTool = async (requestId, payload) => {
  const response = await api.put(`/request-tools/${requestId}`, payload);
  return response.data;
};

export const deleteRequestTool = async (requestId) => {
  const response = await api.delete(`/request-tools/${requestId}`);
  return response.data;
};

export const executeRequestTool = async (requestId) => {
  const response = await api.post(`/request-tools/${requestId}/execute`);
  return response.data;
};

// Conversations
export const getConversations = async () => {
  const response = await api.get('/conversations');
  return response.data;
};

export const getConversation = async (configId) => {
  const response = await api.get(`/conversations/${configId}`);
  return response.data;
};

export const createConversation = async (payload) => {
  const response = await api.post('/conversations', payload);
  return response.data;
};

export const updateConversation = async (configId, payload) => {
  const response = await api.put(`/conversations/${configId}`, payload);
  return response.data;
};

export const deleteConversation = async (configId) => {
  const response = await api.delete(`/conversations/${configId}`);
  return response.data;
};

export const startConversation = async (payload) => {
  const response = await api.post('/conversations/start', payload);
  return response.data;
};

export const continueConversation = async (payload) => {
  const response = await api.post('/conversations/continue', payload);
  return response.data;
};

export const getConversationHistory = async (sessionId) => {
  const response = await api.get(`/conversations/history/${sessionId}`);
  return response.data;
};

export const listSavedConversations = async () => {
  const response = await api.get('/conversations/saved');
  return response.data;
};

export const getSavedConversationContent = async (filename) => {
  const response = await api.get(`/conversations/saved/${encodeURIComponent(filename)}`);
  return response.data;
};

// Image Generation
export const generateImage = async (payload) => {
  const response = await api.post('/images/generate', payload);
  return response.data;
};

export const polishImagePrompt = async (payload) => {
  const response = await api.post('/images/polish-prompt', payload);
  return response.data;
};

export const getGeneratedImages = async () => {
  const response = await api.get('/images');
  return response.data;
};

export const deleteGeneratedImage = async (filename) => {
  const response = await api.delete(`/images/${encodeURIComponent(filename)}`);
  return response.data;
};

// Browser Automation
export const executeBrowserAutomation = async (payload) => {
  const response = await api.post('/browser-automation/execute', payload);
  return response.data;
};

// Export all functions
const apiService = {
  getStatus,
  getRAGCollections,
  addRAGData,
  validateRAGData,
  queryRAGCollection,
  deleteRAGCollection,
  getAgents,
  getAgent,
  createAgent,
  updateAgent,
  deleteAgent,
  runAgent,
  getTools,
  updateToolConfig,
  getModels,
  getProviders,
  startMCPServer,
  getCustomizations,
  createCustomization,
  updateCustomization,
  deleteCustomization,
  queryCustomization,
  getDialogues,
  getDialogue,
  createDialogue,
  updateDialogue,
  deleteDialogue,
  startDialogue,
  continueDialogue,
  crawlWebsite,
  getCrawlerProfiles,
  getCrawlerProfile,
  createCrawlerProfile,
  updateCrawlerProfile,
  deleteCrawlerProfile,
  executeCrawlerProfile,
  getDBTools,
  getDBTool,
  createDBTool,
  updateDBTool,
  deleteDBTool,
  previewDBTool,
  executeDBTool,
  getRequestTools,
  getRequestTool,
  createRequestTool,
  updateRequestTool,
  deleteRequestTool,
  executeRequestTool,
  
  // Flows
  getFlows: async () => {
    const response = await api.get('/flows');
    return response.data;
  },
  getFlow: async (flowId) => {
    const response = await api.get(`/flows/${flowId}`);
    return response.data;
  },
  createFlow: async (payload) => {
    const response = await api.post('/flows', payload);
    return response.data;
  },
  updateFlow: async (flowId, payload) => {
    const response = await api.put(`/flows/${flowId}`, payload);
    return response.data;
  },
  deleteFlow: async (flowId) => {
    const response = await api.delete(`/flows/${flowId}`);
    return response.data;
  },
  executeFlow: async (flowId, payload) => {
    const response = await api.post(`/flows/${flowId}/execute`, payload);
    return response.data;
  },
  // Special Flow 1
  getSpecialFlows1: async () => {
    const response = await api.get('/special-flows-1');
    return response.data;
  },
  getSpecialFlow1: async (flowId) => {
    const response = await api.get(`/special-flows-1/${flowId}`);
    return response.data;
  },
  createSpecialFlow1: async (payload) => {
    const response = await api.post('/special-flows-1', payload);
    return response.data;
  },
  updateSpecialFlow1: async (flowId, payload) => {
    const response = await api.put(`/special-flows-1/${flowId}`, payload);
    return response.data;
  },
  deleteSpecialFlow1: async (flowId) => {
    const response = await api.delete(`/special-flows-1/${flowId}`);
    return response.data;
  },
  executeSpecialFlow1: async (flowId, payload) => {
    const response = await api.post(`/special-flows-1/${flowId}/execute`, payload);
    return response.data;
  },
  // Conversations
  getConversations,
  getConversation,
  createConversation,
  updateConversation,
  deleteConversation,
  startConversation,
  continueConversation,
  getConversationHistory,
  listSavedConversations,
  getSavedConversationContent,
  // Image Generation
  generateImage,
  polishImagePrompt,
  getGeneratedImages,
  deleteGeneratedImage,
  // Browser Automation
  executeBrowserAutomation,
};

export default apiService; 