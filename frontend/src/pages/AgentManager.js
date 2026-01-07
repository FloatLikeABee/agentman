import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Grid,
  IconButton,
  Switch,
  FormControlLabel,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  CheckCircle as CheckCircleIcon,
  Edit as EditIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import ReactMarkdown from 'react-markdown';
import api from '../services/api';

const AgentManager = () => {
  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openRunDialog, setOpenRunDialog] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [editingAgentId, setEditingAgentId] = useState(null);
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState({ open: false, agentId: null });
  const [queryText, setQueryText] = useState('');
  const [agentResponse, setAgentResponse] = useState('');
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    agent_type: 'rag',
    llm_provider: 'gemini',
    model_name: '',
    temperature: 0.7,
    max_tokens: 8192,
    rag_collections: [],
    tools: [],
    system_prompt: '',
    system_prompt_data: '',
    is_active: true,
  });

  const queryClient = useQueryClient();
  const { data: agents, isLoading } = useQuery('agents', api.getAgents);
  const { data: models } = useQuery('models', api.getModels);
  const { data: collections } = useQuery('collections', api.getRAGCollections);
  const { data: tools } = useQuery('tools', api.getTools);
  const { data: providersData } = useQuery('providers', api.getProviders);

  const createAgentMutation = useMutation(api.createAgent, {
    onSuccess: () => {
      queryClient.invalidateQueries('agents');
      setOpenCreateDialog(false);
      setEditingAgentId(null);
      setFormData({
        name: '',
        description: '',
        agent_type: 'rag',
        llm_provider: 'gemini',
        model_name: '',
        temperature: 0.7,
        max_tokens: 8192,
        rag_collections: [],
        tools: [],
        system_prompt: '',
        is_active: true,
      });
    },
  });

  const updateAgentMutation = useMutation(
    ({ agentId, config }) => api.updateAgent(agentId, config),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agents');
        setOpenCreateDialog(false);
        setEditingAgentId(null);
        setFormData({
          name: '',
          description: '',
          agent_type: 'rag',
          llm_provider: 'gemini',
          model_name: '',
          temperature: 0.7,
          max_tokens: 8192,
          rag_collections: [],
          tools: [],
          system_prompt: '',
          is_active: true,
        });
      },
    }
  );

  const deleteAgentMutation = useMutation(api.deleteAgent, {
    onSuccess: () => {
      queryClient.invalidateQueries('agents');
      setDeleteConfirmDialog({ open: false, agentId: null });
    },
  });

  const runAgentStreamMutation = useMutation(
    ({ agentId, query, context }) => api.runAgent(agentId, query, context),
    {
      onSuccess: (data) => {
        setAgentResponse(data.response);
      },
    }
  );

  // Clear query and response when dialog opens or selected agent changes
  useEffect(() => {
    if (openRunDialog && selectedAgent) {
      setQueryText('');
      setAgentResponse('');
    }
  }, [openRunDialog, selectedAgent]);

  const handleCreateAgent = () => {
    if (editingAgentId) {
      updateAgentMutation.mutate({ agentId: editingAgentId, config: formData });
    } else {
      createAgentMutation.mutate(formData);
    }
  };

  const handleEditAgent = async (agentId) => {
    try {
      // Try to get full agent details from API
      const agent = await api.getAgent(agentId);
      let config = null;
      
      // Handle different response structures
      if (agent && agent.config) {
        config = agent.config;
      } else if (agent && agent.name) {
        // If agent data is directly the config
        config = agent;
      } else {
        // Fallback: try to find agent in the list
        const agentFromList = agents?.find(a => a.id === agentId);
        if (agentFromList) {
          // We have limited data from list, need to fetch full details
          // But if API doesn't return config, we'll use what we have
          config = agentFromList;
        }
      }
      
      if (config) {
        setFormData({
          name: config.name || '',
          description: config.description || '',
          agent_type: config.agent_type || 'rag',
          llm_provider: config.llm_provider || 'gemini',
          model_name: config.model_name || '',
          temperature: config.temperature || 0.7,
          max_tokens: config.max_tokens || 8192,
          rag_collections: config.rag_collections || [],
          tools: config.tools || [],
          system_prompt: config.system_prompt || '',
          system_prompt_data: config.system_prompt_data || '',
          is_active: config.is_active !== undefined ? config.is_active : true,
        });
        setEditingAgentId(agentId);
        setOpenCreateDialog(true);
      } else {
        throw new Error('Agent configuration not found');
      }
    } catch (error) {
      console.error('Error loading agent for editing:', error);
      alert('Failed to load agent details. Please try again.');
    }
  };

  const handleDeleteAgent = (agentId) => {
    setDeleteConfirmDialog({ open: true, agentId });
  };

  const handleDeleteConfirm = () => {
    deleteAgentMutation.mutate(deleteConfirmDialog.agentId);
  };

  const handleRunAgent = () => {
    if (selectedAgent && queryText) {
      runAgentStreamMutation.mutate({
        agentId: selectedAgent.id,
        query: queryText,
      });
    }
  };

  if (isLoading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Agent Manager</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpenCreateDialog(true)}
        >
          Create Agent
        </Button>
      </Box>

      {/* Agents List */}
      <Grid container spacing={3}>
        {agents?.map((agent) => (
          <Grid item xs={12} md={6} lg={4} key={agent.id}>
            <Card sx={{ boxShadow: 2, transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' }, height: '100%' }}>
              <CardContent sx={{ p: 3, minHeight: 220, display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2, flexGrow: 1 }}>
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="h6" sx={{ mb: 1 }}>{agent.name}</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {agent.description}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      <Chip
                        label={agent.agent_type}
                        size="small"
                        color="primary"
                      />
                      <Chip
                        label={agent.model_name}
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        label={agent.is_active ? 'Active' : 'Inactive'}
                        size="small"
                        color={agent.is_active ? 'success' : 'default'}
                      />
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    <IconButton
                      size="small"
                      color="primary"
                      onClick={() => {
                        setQueryText(''); // Clear previous query
                        setAgentResponse(''); // Clear previous response
                        setSelectedAgent(agent);
                        setOpenRunDialog(true);
                      }}
                      sx={{ bgcolor: 'primary.light', '&:hover': { bgcolor: 'primary.main', color: 'white' } }}
                      title="Run Agent"
                    >
                      <RunIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="secondary"
                      onClick={() => handleEditAgent(agent.id)}
                      sx={{ bgcolor: 'secondary.light', '&:hover': { bgcolor: 'secondary.main', color: 'white' } }}
                      title="Edit Agent"
                    >
                      <EditIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleDeleteAgent(agent.id)}
                      sx={{ bgcolor: 'error.light', '&:hover': { bgcolor: 'error.main', color: 'white' } }}
                      title="Delete Agent"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Collections: {agent.rag_collections?.length || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Tools: {agent.tools?.length || 0}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Create/Edit Agent Dialog */}
      <Dialog 
        open={openCreateDialog} 
        onClose={() => {
          setOpenCreateDialog(false);
          setEditingAgentId(null);
          setFormData({
            name: '',
            description: '',
            agent_type: 'rag',
            llm_provider: 'gemini',
            model_name: '',
            temperature: 0.7,
            max_tokens: 8192,
            rag_collections: [],
            tools: [],
            system_prompt: '',
            is_active: true,
          });
        }} 
        maxWidth="md" 
        fullWidth
      >
        <DialogTitle sx={{ pb: 1 }}>
          {editingAgentId ? 'Edit Agent' : 'Create New Agent'}
        </DialogTitle>
        <DialogContent sx={{ pt: 1 }}>
          {(createAgentMutation.isError || updateAgentMutation.isError) && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {editingAgentId 
                ? 'Failed to update agent. Please check your inputs and try again.'
                : 'Failed to create agent. Please check your inputs and try again.'}
            </Alert>
          )}
          <Grid container spacing={3}>
            {/* Basic Information */}
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom color="primary">
                Basic Information
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Agent Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                required
                error={!formData.name.trim()}
                helperText={!formData.name.trim() ? 'Name is required' : ''}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Agent Type</InputLabel>
                <Select
                  value={formData.agent_type}
                  onChange={(e) => setFormData({ ...formData, agent_type: e.target.value })}
                >
                  <MenuItem value="rag">RAG</MenuItem>
                  <MenuItem value="tool">Tool</MenuItem>
                  <MenuItem value="hybrid">Hybrid</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                multiline
                rows={2}
              />
            </Grid>

            {/* Model Settings */}
            <Grid item xs={12} sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom color="primary">
                Model Settings
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>LLM Provider</InputLabel>
                <Select
                  value={formData.llm_provider}
                  onChange={(e) => setFormData({ ...formData, llm_provider: e.target.value })}
                >
                  {providersData?.providers?.map((provider) => (
                    <MenuItem key={provider} value={provider}>
                      {provider.charAt(0).toUpperCase() + provider.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Model</InputLabel>
                <Select
                  value={formData.model_name}
                  onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                >
                  {models?.map((model) => (
                    <MenuItem key={model.name} value={model.name}>
                      {model.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Temperature"
                value={formData.temperature}
                onChange={(e) => setFormData({ ...formData, temperature: parseFloat(e.target.value) })}
                inputProps={{ min: 0, max: 2, step: 0.1 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Max Tokens"
                value={formData.max_tokens}
                onChange={(e) => setFormData({ ...formData, max_tokens: parseInt(e.target.value) })}
                inputProps={{ min: 1, max: 32768 }}
                helperText="Maximum 32,768 tokens"
              />
            </Grid>

            {/* Capabilities */}
            <Grid item xs={12} sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom color="primary">
                Capabilities
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>RAG Collections</InputLabel>
                <Select
                  multiple
                  value={formData.rag_collections}
                  onChange={(e) => setFormData({ ...formData, rag_collections: e.target.value })}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {collections?.map((collection) => (
                    <MenuItem key={collection.name} value={collection.name}>
                      {collection.name} ({collection.count} documents)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Tools</InputLabel>
                <Select
                  multiple
                  value={formData.tools}
                  onChange={(e) => setFormData({ ...formData, tools: e.target.value })}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {tools?.map((tool) => (
                    <MenuItem key={tool.id} value={tool.name}>
                      {tool.name} - {tool.description}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.is_active}
                    onChange={(e) => setFormData({ ...formData, is_active: e.target.checked })}
                  />
                }
                label="Active"
              />
            </Grid>

            {/* Advanced Settings */}
            <Grid item xs={12} sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom color="primary">
                Advanced Settings
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="System Prompt"
                value={formData.system_prompt}
                onChange={(e) => setFormData({ ...formData, system_prompt: e.target.value })}
                placeholder="Enter system prompt for the agent... Use {data} placeholder to inject data from flows."
                helperText="Use {data} placeholder to inject data from flows into the system prompt"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="System Prompt Data (Optional)"
                value={formData.system_prompt_data || ''}
                onChange={(e) => setFormData({ ...formData, system_prompt_data: e.target.value })}
                placeholder="Data to inject into system prompt (replaces {data} placeholder). Leave empty if using flows."
                helperText="This data will replace {data} in the system prompt. Leave empty if data will come from flows."
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions sx={{ p: 3, pt: 1 }}>
          <Button 
            onClick={() => {
              setOpenCreateDialog(false);
              setEditingAgentId(null);
              setFormData({
                name: '',
                description: '',
                agent_type: 'rag',
                llm_provider: 'gemini',
                model_name: '',
                temperature: 0.7,
                max_tokens: 8192,
                rag_collections: [],
                tools: [],
                system_prompt: '',
                system_prompt_data: '',
                is_active: true,
              });
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleCreateAgent}
            variant="contained"
            disabled={
              (createAgentMutation.isLoading || updateAgentMutation.isLoading) || 
              !formData.name.trim()
            }
          >
            {editingAgentId 
              ? (updateAgentMutation.isLoading ? 'Updating...' : 'Update Agent')
              : (createAgentMutation.isLoading ? 'Creating...' : 'Create Agent')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Run Agent Dialog */}
      <Dialog
        open={openRunDialog}
        onClose={() => {
          setOpenRunDialog(false);
          setQueryText(''); // Clear query when closing
          setAgentResponse(''); // Clear response when closing
        }}
        maxWidth="lg"
        fullWidth
        sx={{
          '& .MuiDialog-paper': {
            height: '80vh',
            maxHeight: '600px',
          }
        }}
      >
        <DialogTitle sx={{ pb: 1 }}>Run Agent: {selectedAgent?.name}</DialogTitle>
        <DialogContent sx={{ pt: 1, display: 'flex', flexDirection: 'column' }}>
          {runAgentStreamMutation.isError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to run agent. Please try again.
            </Alert>
          )}

          {/* Query Input Section */}
          <Box sx={{ mb: 3, flexShrink: 0 }}>
            <TextField
              fullWidth
              label="Query"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              multiline
              rows={3}
              placeholder="Enter your query for the agent..."
              sx={{ mb: 2 }}
            />
            <Button
              variant="contained"
              onClick={handleRunAgent}
              disabled={runAgentStreamMutation.isLoading || !queryText.trim()}
              size="large"
              fullWidth
            >
              {runAgentStreamMutation.isLoading ? 'Running Agent...' : 'Run Agent'}
            </Button>
          </Box>

          {/* Response Section */}
          {agentResponse && (
            <Box sx={{ flex: 1, minHeight: 0 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CheckCircleIcon color="success" />
                Response
              </Typography>
              <Box
                sx={{
                  mt: 1,
                  p: 2,
                  bgcolor: 'grey.50',
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'grey.200',
                  flex: 1,
                  overflow: 'hidden',
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                <Box
                  sx={{
                    flex: 1,
                    overflowY: 'auto',
                    '&::-webkit-scrollbar': {
                      width: '8px',
                    },
                    '&::-webkit-scrollbar-track': {
                      bgcolor: 'grey.100',
                      borderRadius: '4px',
                    },
                    '&::-webkit-scrollbar-thumb': {
                      bgcolor: 'grey.400',
                      borderRadius: '4px',
                      '&:hover': {
                        bgcolor: 'grey.500',
                      },
                    },
                  }}
                >
                  <Box sx={{ lineHeight: 1.6 }}>
                    <ReactMarkdown
                      components={{
                        p: ({ children }) => (
                          <Typography variant="body1" sx={{ mb: 1 }}>
                            {children}
                          </Typography>
                        ),
                        strong: ({ children }) => (
                          <Typography component="strong" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                            {children}
                          </Typography>
                        ),
                        em: ({ children }) => (
                          <Typography component="em" sx={{ fontStyle: 'italic' }}>
                            {children}
                          </Typography>
                        ),
                        h1: ({ children }) => (
                          <Typography variant="h5" sx={{ mb: 2, fontWeight: 'bold', color: 'primary.main' }}>
                            {children}
                          </Typography>
                        ),
                        h2: ({ children }) => (
                          <Typography variant="h6" sx={{ mb: 1, fontWeight: 'bold', color: 'primary.main' }}>
                            {children}
                          </Typography>
                        ),
                        h3: ({ children }) => (
                          <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>
                            {children}
                          </Typography>
                        ),
                        ul: ({ children }) => (
                          <Box component="ul" sx={{ pl: 2, mb: 1 }}>
                            {children}
                          </Box>
                        ),
                        ol: ({ children }) => (
                          <Box component="ol" sx={{ pl: 2, mb: 1 }}>
                            {children}
                          </Box>
                        ),
                        li: ({ children }) => (
                          <Typography component="li" variant="body1" sx={{ mb: 0.5 }}>
                            {children}
                          </Typography>
                        ),
                        code: ({ children }) => (
                          <Box
                            component="code"
                            sx={{
                              bgcolor: 'grey.200',
                              px: 0.5,
                              py: 0.25,
                              borderRadius: 0.5,
                              fontFamily: 'monospace',
                              fontSize: '0.875rem',
                            }}
                          >
                            {children}
                          </Box>
                        ),
                        pre: ({ children }) => (
                          <Box
                            component="pre"
                            sx={{
                              bgcolor: 'grey.200',
                              p: 1,
                              borderRadius: 1,
                              overflowX: 'auto',
                              fontFamily: 'monospace',
                              fontSize: '0.875rem',
                              mb: 1,
                            }}
                          >
                            {children}
                          </Box>
                        ),
                      }}
                    >
                      {agentResponse}
                    </ReactMarkdown>
                  </Box>
                </Box>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ p: 3, pt: 1, flexShrink: 0 }}>
          <Button onClick={() => setOpenRunDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmDialog.open}
        onClose={() => setDeleteConfirmDialog({ open: false, agentId: null })}
      >
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this agent? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmDialog({ open: false, agentId: null })}>
            Cancel
          </Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AgentManager; 