import React, { useState } from 'react';
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
  Grid,
  IconButton,
  Alert,
  Chip,
  Autocomplete,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Paper,
  Divider,
} from '@mui/material';
import ReactMarkdown from 'react-markdown';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  Edit as EditIcon,
  Send as SendIcon,
  History as HistoryIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';
import SystemPromptInput from './SystemPromptInput';

const Conversation = () => {
  const queryClient = useQueryClient();
  const { data: conversations = [], isLoading } = useQuery('conversations', api.getConversations, { staleTime: 5 * 60 * 1000 });
  const { data: models = [] } = useQuery('models', api.getModels, { enabled: true, staleTime: 5 * 60 * 1000 });
  const { data: providersData } = useQuery('providers', api.getProviders, { enabled: true, staleTime: 5 * 60 * 1000 });
  const { data: collections = [] } = useQuery('collections', api.getRAGCollections, { staleTime: 5 * 60 * 1000 });
  const { data: dbTools = [] } = useQuery('db-tools', api.getDBTools, { staleTime: 5 * 60 * 1000 });
  const { data: requestTools = [] } = useQuery('request-tools', api.getRequestTools, { staleTime: 5 * 60 * 1000 });
  const { data: savedConversations = [] } = useQuery('saved-conversations', api.listSavedConversations, { staleTime: 5 * 60 * 1000 });

  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [openHistoryDialog, setOpenHistoryDialog] = useState(false);
  const [editingConfigId, setEditingConfigId] = useState(null);
  const [selectedConfig, setSelectedConfig] = useState(null);
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState({ open: false, configId: null });
  const [createForm, setCreateForm] = useState({
    name: '',
    description: '',
    config: {
      model1_config: {
        provider: 'gemini',
        model_name: '',
        system_prompt: '',
        rag_collection: '',
      },
      model2_config: {
        provider: 'gemini',
        model_name: '',
        system_prompt: '',
        rag_collection: '',
      },
      max_turns: 10,
    },
  });

  // Conversation execution state
  const [sessionId, setSessionId] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [currentTopic, setCurrentTopic] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [conversationError, setConversationError] = useState('');
  const [userMessage, setUserMessage] = useState('');
  const [requestLogs, setRequestLogs] = useState([]);
  const [viewConversationModal, setViewConversationModal] = useState({ open: false, filename: null, content: null });

  const createMutation = useMutation(api.createConversation, {
    onSuccess: () => {
      queryClient.invalidateQueries('conversations');
      setOpenCreateDialog(false);
      resetForm();
    },
  });

  const updateMutation = useMutation(
    ({ configId, payload }) => api.updateConversation(configId, payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('conversations');
        setOpenEditDialog(false);
        setEditingConfigId(null);
        resetForm();
      },
    }
  );

  const deleteMutation = useMutation(api.deleteConversation, {
    onSuccess: () => {
      queryClient.invalidateQueries('conversations');
      setDeleteConfirmDialog({ open: false, configId: null });
    },
  });

  const startMutation = useMutation(api.startConversation, {
    onSuccess: (data) => {
      setSessionId(data.session_id);
      setConversationHistory(data.conversation_history || []);
      setIsRunning(true);
      setConversationError('');
      setCurrentTopic(''); // Clear topic input after starting
      // Add request logs from metadata
      if (data.metadata?.request_logs) {
        setRequestLogs(prev => [...prev, ...data.metadata.request_logs]);
      }
      // Automatically continue if not complete
      if (!data.is_complete && data.session_id) {
        setTimeout(() => {
          continueMutation.mutate({ session_id: data.session_id });
        }, 500);
      }
    },
    onError: (error) => {
      setConversationError(error.response?.data?.detail || 'Failed to start conversation');
      setIsRunning(false);
    },
  });

  const continueMutation = useMutation(api.continueConversation, {
    onSuccess: (data) => {
      setConversationHistory(data.conversation_history || []);
      // Add request logs from metadata
      if (data.metadata?.request_logs) {
        setRequestLogs(prev => [...prev, ...data.metadata.request_logs]);
      }
      if (data.is_complete) {
        setIsRunning(false);
        queryClient.invalidateQueries('saved-conversations');
      } else {
        // Automatically continue to next turn
        if (data.session_id) {
          setTimeout(() => {
            continueMutation.mutate({ session_id: data.session_id });
          }, 500);
        }
      }
      setUserMessage('');
    },
    onError: (error) => {
      setConversationError(error.response?.data?.detail || 'Failed to continue conversation');
      setIsRunning(false);
    },
  });

  const resetForm = () => {
    setCreateForm({
      name: '',
      description: '',
      config: {
        model1_config: {
          provider: 'gemini',
          model_name: '',
          system_prompt: '',
          rag_collection: '',
        },
        model2_config: {
          provider: 'gemini',
          model_name: '',
          system_prompt: '',
          rag_collection: '',
        },
        max_turns: 10,
      },
    });
  };

  const handleCreate = () => {
    createMutation.mutate(createForm);
  };

  const handleEdit = (config) => {
    setEditingConfigId(config.id);
    setCreateForm({
      name: config.name,
      description: config.description || '',
      config: config.config,
    });
    setOpenEditDialog(true);
  };

  const handleUpdate = () => {
    updateMutation.mutate({ configId: editingConfigId, payload: createForm });
  };

  const handleDelete = (configId) => {
    setDeleteConfirmDialog({ open: true, configId });
  };

  const handleDeleteConfirm = () => {
    deleteMutation.mutate(deleteConfirmDialog.configId);
  };

  const handleStart = () => {
    if (!selectedConfig || !currentTopic.trim()) return;
    setRequestLogs([]); // Clear previous logs
    startMutation.mutate({
      config_id: selectedConfig.id,
      topic: currentTopic,
    });
  };

  const handleContinue = () => {
    if (!sessionId) return;
    continueMutation.mutate({
      session_id: sessionId,
      user_message: userMessage || undefined,
    });
  };

  const handleViewHistory = async (sessionId) => {
    try {
      const history = await api.getConversationHistory(sessionId);
      setConversationHistory(history.conversation_history || []);
      setOpenHistoryDialog(true);
    } catch (error) {
      setConversationError(error.response?.data?.detail || 'Failed to load history');
    }
  };

  const providers = providersData?.providers || [];
  const availableModels = models || [];

  if (isLoading) {
    return <Box>Loading...</Box>;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Conversation Manager</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            resetForm();
            setEditingConfigId(null);
            setOpenCreateDialog(true);
          }}
        >
          Create Configuration
        </Button>
      </Box>

      {/* Configurations List */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {conversations.map((config) => (
          <Grid item xs={12} md={6} lg={4} key={config.id}>
            <Card sx={{ boxShadow: 2 }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      {config.name}
                    </Typography>
                    {config.description && (
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {config.description}
                      </Typography>
                    )}
                  </Box>
                  <Box>
                    <IconButton size="small" onClick={() => handleEdit(config)}>
                      <EditIcon />
                    </IconButton>
                    <IconButton size="small" onClick={() => handleDelete(config.id)} color="error">
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </Box>
                <Box sx={{ mb: 2 }}>
                  <Chip label={`${config.config.model1_config.model_name} + ${config.config.model2_config.model_name}`} size="small" sx={{ mr: 1 }} />
                  <Chip label={`Max ${config.config.max_turns} turns`} size="small" />
                </Box>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<RunIcon />}
                  onClick={() => {
                    setSelectedConfig(config);
                    setCurrentTopic('');
                    setSessionId(null);
                    setConversationHistory([]);
                    setIsRunning(false);
                    setConversationError('');
                  }}
                >
                  Start Conversation
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
        {conversations.length === 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                  No conversation configurations yet. Create one to get started.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Conversation Execution Panel */}
      {selectedConfig && (
        <Card sx={{ boxShadow: 2, mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Conversation: {selectedConfig.name}
            </Typography>
            {conversationError && (
              <Alert severity="error" sx={{ mb: 2 }} onClose={() => setConversationError('')}>
                {conversationError}
              </Alert>
            )}
            {!isRunning ? (
              <Box>
                <TextField
                  fullWidth
                  label="Topic / Initial Prompt"
                  value={currentTopic}
                  onChange={(e) => setCurrentTopic(e.target.value)}
                  multiline
                  rows={3}
                  sx={{ mb: 2 }}
                  placeholder="Enter a topic or prompt to start the conversation..."
                />
                <Button
                  variant="contained"
                  startIcon={<RunIcon />}
                  onClick={handleStart}
                  disabled={!currentTopic.trim() || startMutation.isLoading}
                >
                  {startMutation.isLoading ? 'Starting...' : 'Start Conversation'}
                </Button>
              </Box>
            ) : (
              <Box>
                {/* Conversation Messages - styled like request logs */}
                {conversationHistory.length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold', color: 'primary.main' }}>
                      Conversation
                    </Typography>
                    <Box
                      sx={{
                        p: 2,
                        bgcolor: 'background.default',
                        border: '1px solid',
                        borderColor: 'primary.main',
                        borderOpacity: 0.2,
                        borderRadius: 1,
                        maxHeight: 500,
                        overflowY: 'auto',
                        '&::-webkit-scrollbar': { width: '8px' },
                        '&::-webkit-scrollbar-track': { bgcolor: 'background.paper', borderRadius: '4px' },
                        '&::-webkit-scrollbar-thumb': { bgcolor: 'primary.main', bgcolorOpacity: 0.5, borderRadius: '4px' },
                      }}
                    >
                      {conversationHistory.map((msg, idx) => (
                        <Box
                          key={idx}
                          sx={{
                            mb: 2,
                            p: 1.5,
                            bgcolor: 'background.paper',
                            border: '1px solid',
                            borderColor: msg.role === 'user' ? 'primary.main' : 'primary.main',
                            borderOpacity: 0.3,
                            borderRadius: 1,
                          }}
                        >
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography variant="caption" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                              Turn {msg.turn_number} • {msg.role === 'user' ? 'USER' : msg.role === 'model1' ? `AI MODEL 1 (${selectedConfig.config.model1_config.model_name})` : `AI MODEL 2 (${selectedConfig.config.model2_config.model_name})`}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {new Date(msg.timestamp).toLocaleTimeString()}
                            </Typography>
                          </Box>
                          <Box
                            sx={{
                              mt: 1,
                              p: 1,
                              bgcolor: 'background.default',
                              borderRadius: 0.5,
                              border: '1px solid',
                              borderColor: 'primary.main',
                              borderOpacity: 0.1,
                            }}
                          >
                            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                              {msg.content}
                            </Typography>
                          </Box>
                        </Box>
                      ))}
                    </Box>
                  </Box>
                )}
                
                {/* Request/Response Logs */}
                {requestLogs.length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold', color: 'primary.main' }}>
                      AI Model Requests & Responses
                    </Typography>
                    <Box
                      sx={{
                        p: 2,
                        bgcolor: 'background.default',
                        border: '1px solid',
                        borderColor: 'primary.main',
                        borderOpacity: 0.2,
                        borderRadius: 1,
                        maxHeight: 300,
                        overflowY: 'auto',
                        '&::-webkit-scrollbar': { width: '8px' },
                        '&::-webkit-scrollbar-track': { bgcolor: 'background.paper', borderRadius: '4px' },
                        '&::-webkit-scrollbar-thumb': { bgcolor: 'primary.main', bgcolorOpacity: 0.5, borderRadius: '4px' },
                      }}
                    >
                      {requestLogs.map((log, idx) => (
                        <Box
                          key={idx}
                          sx={{
                            mb: 2,
                            p: 1.5,
                            bgcolor: 'background.paper',
                            border: '1px solid',
                            borderColor: log.error ? 'error.main' : 'primary.main',
                            borderOpacity: 0.3,
                            borderRadius: 1,
                          }}
                        >
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography variant="caption" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                              Turn {log.turn} • {log.model}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {new Date(log.request_timestamp).toLocaleTimeString()}
                            </Typography>
                          </Box>
                          <Typography variant="caption" sx={{ display: 'block', mb: 0.5, color: 'text.secondary' }}>
                            Provider: {log.provider} • Model: {log.model_name}
                            {log.rag_collection && ` • RAG: ${log.rag_collection}`}
                          </Typography>
                          {log.error ? (
                            <Alert severity="error" sx={{ mt: 1 }}>
                              Error: {log.error}
                            </Alert>
                          ) : (
                            <>
                              <Typography variant="caption" sx={{ display: 'block', mb: 0.5, color: 'text.secondary' }}>
                                Prompt: {log.prompt_length} chars • Response: {log.response_length} chars
                              </Typography>
                              <Box
                                sx={{
                                  mt: 1,
                                  p: 1,
                                  bgcolor: 'background.default',
                                  borderRadius: 0.5,
                                  border: '1px solid',
                                  borderColor: 'primary.main',
                                  borderOpacity: 0.1,
                                }}
                              >
                                <Typography variant="caption" sx={{ fontFamily: 'monospace', fontSize: '0.75rem', whiteSpace: 'pre-wrap' }}>
                                  {log.response_preview}
                                </Typography>
                              </Box>
                              {log.response_timestamp && (
                                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary', fontStyle: 'italic' }}>
                                  Response received: {new Date(log.response_timestamp).toLocaleTimeString()}
                                </Typography>
                              )}
                            </>
                          )}
                        </Box>
                      ))}
                    </Box>
                  </Box>
                )}
                
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    label="User Message (Optional)"
                    value={userMessage}
                    onChange={(e) => setUserMessage(e.target.value)}
                    placeholder="Enter a message to inject into the conversation..."
                    size="small"
                  />
                  <Button
                    variant="contained"
                    startIcon={<SendIcon />}
                    onClick={handleContinue}
                    disabled={continueMutation.isLoading}
                  >
                    {continueMutation.isLoading ? 'Sending...' : 'Continue'}
                  </Button>
                </Box>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* Saved Conversations */}
      {savedConversations.length > 0 && (
        <Card sx={{ boxShadow: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Saved Conversations
            </Typography>
            <Grid container spacing={2}>
              {savedConversations.slice(0, 10).map((conv) => (
                <Grid item xs={12} sm={6} md={4} key={conv.filename}>
                  <Paper 
                    sx={{ 
                      p: 2, 
                      border: '1px solid', 
                      borderColor: 'primary.main', 
                      borderOpacity: 0.3,
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      '&:hover': {
                        borderColor: 'primary.main',
                        borderOpacity: 0.6,
                        bgcolor: 'action.hover',
                        transform: 'translateY(-2px)',
                        boxShadow: 2,
                      }
                    }}
                    onClick={async () => {
                      try {
                        const data = await api.getSavedConversationContent(conv.filename);
                        setViewConversationModal({ open: true, filename: conv.filename, content: data.content });
                      } catch (error) {
                        console.error('Error loading conversation:', error);
                      }
                    }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: 'medium', mb: 1 }}>
                      {conv.filename}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(conv.created_at).toLocaleString()}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* View Conversation Modal */}
      <Dialog 
        open={viewConversationModal.open} 
        onClose={() => setViewConversationModal({ open: false, filename: null, content: null })} 
        maxWidth="lg" 
        fullWidth
      >
        <DialogTitle>
          {viewConversationModal.filename}
        </DialogTitle>
        <DialogContent>
          <Box
            sx={{
              mt: 2,
              p: 2,
              bgcolor: 'background.default',
              border: '1px solid',
              borderColor: 'primary.main',
              borderOpacity: 0.2,
              borderRadius: 1,
              maxHeight: '70vh',
              overflowY: 'auto',
              '&::-webkit-scrollbar': { width: '8px' },
              '&::-webkit-scrollbar-track': { bgcolor: 'background.paper', borderRadius: '4px' },
              '&::-webkit-scrollbar-thumb': { bgcolor: 'primary.main', bgcolorOpacity: 0.5, borderRadius: '4px' },
            }}
          >
            {viewConversationModal.content ? (
              <Box
                sx={{
                  '& h1, & h2, & h3, & h4, & h5, & h6': {
                    color: 'primary.main',
                    mt: 2,
                    mb: 1,
                  },
                  '& p': {
                    mb: 1,
                  },
                  '& pre': {
                    bgcolor: 'background.paper',
                    p: 2,
                    borderRadius: 1,
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                  },
                  '& code': {
                    bgcolor: 'background.paper',
                    px: 0.5,
                    borderRadius: 0.5,
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                  },
                  '& blockquote': {
                    borderLeft: '3px solid',
                    borderColor: 'primary.main',
                    pl: 2,
                    ml: 0,
                    fontStyle: 'italic',
                  },
                  '& ul, & ol': {
                    pl: 3,
                    mb: 1,
                  },
                  '& hr': {
                    borderColor: 'primary.main',
                    borderOpacity: 0.3,
                    my: 2,
                  },
                }}
              >
                <ReactMarkdown>{viewConversationModal.content}</ReactMarkdown>
              </Box>
            ) : (
              <Typography>Loading...</Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewConversationModal({ open: false, filename: null, content: null })}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create/Edit Dialog */}
      <Dialog open={openCreateDialog || openEditDialog} onClose={() => { setOpenCreateDialog(false); setOpenEditDialog(false); }} maxWidth="md" fullWidth>
        <DialogTitle>{editingConfigId ? 'Edit Configuration' : 'Create Configuration'}</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Name"
              value={createForm.name}
              onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Description"
              value={createForm.description}
              onChange={(e) => setCreateForm({ ...createForm, description: e.target.value })}
              multiline
              rows={2}
              sx={{ mb: 2 }}
            />
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" gutterBottom>
              Model 1 Configuration
            </Typography>
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Provider</InputLabel>
                  <Select
                    value={createForm.config.model1_config.provider}
                    onChange={(e) => setCreateForm({ 
                      ...createForm, 
                      config: { 
                        ...createForm.config, 
                        model1_config: { ...createForm.config.model1_config, provider: e.target.value }
                      } 
                    })}
                    label="Provider"
                  >
                    {providers.map((p) => (
                      <MenuItem key={p} value={p}>{p}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Autocomplete
                  freeSolo
                  options={availableModels.filter(m => m.name).map(m => m.name)}
                  value={createForm.config.model1_config.model_name}
                  onChange={(e, newValue) => setCreateForm({ 
                    ...createForm, 
                    config: { 
                      ...createForm.config, 
                      model1_config: { ...createForm.config.model1_config, model_name: newValue || '' }
                    } 
                  })}
                  renderInput={(params) => <TextField {...params} label="Model Name" />}
                />
              </Grid>
            </Grid>
            <Autocomplete
              freeSolo
              options={collections.map((col) => col.name)}
              value={createForm.config.model1_config.rag_collection || ''}
              onChange={(event, newValue) => {
                setCreateForm({ 
                  ...createForm, 
                  config: { 
                    ...createForm.config, 
                    model1_config: { ...createForm.config.model1_config, rag_collection: newValue || '' }
                  } 
                });
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Model 1 RAG Collection (optional)"
                  helperText="Optional RAG collection for Model 1"
                  sx={{ mb: 2 }}
                />
              )}
            />
            <SystemPromptInput
              fullWidth
              label="Model 1 System Prompt"
              value={createForm.config.model1_config.system_prompt}
              onChange={(e) => setCreateForm({ 
                ...createForm, 
                config: { 
                  ...createForm.config, 
                  model1_config: { ...createForm.config.model1_config, system_prompt: e.target.value }
                } 
              })}
              rows={4}
              sx={{ mb: 2 }}
              placeholder="Enter system prompt for Model 1..."
            />
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" gutterBottom>
              Model 2 Configuration
            </Typography>
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Provider</InputLabel>
                  <Select
                    value={createForm.config.model2_config.provider}
                    onChange={(e) => setCreateForm({ 
                      ...createForm, 
                      config: { 
                        ...createForm.config, 
                        model2_config: { ...createForm.config.model2_config, provider: e.target.value }
                      } 
                    })}
                    label="Provider"
                  >
                    {providers.map((p) => (
                      <MenuItem key={p} value={p}>{p}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Autocomplete
                  freeSolo
                  options={availableModels.filter(m => m.name).map(m => m.name)}
                  value={createForm.config.model2_config.model_name}
                  onChange={(e, newValue) => setCreateForm({ 
                    ...createForm, 
                    config: { 
                      ...createForm.config, 
                      model2_config: { ...createForm.config.model2_config, model_name: newValue || '' }
                    } 
                  })}
                  renderInput={(params) => <TextField {...params} label="Model Name" />}
                />
              </Grid>
            </Grid>
            <Autocomplete
              freeSolo
              options={collections.map((col) => col.name)}
              value={createForm.config.model2_config.rag_collection || ''}
              onChange={(event, newValue) => {
                setCreateForm({ 
                  ...createForm, 
                  config: { 
                    ...createForm.config, 
                    model2_config: { ...createForm.config.model2_config, rag_collection: newValue || '' }
                  } 
                });
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Model 2 RAG Collection (optional)"
                  helperText="Optional RAG collection for Model 2"
                  sx={{ mb: 2 }}
                />
              )}
            />
            <SystemPromptInput
              fullWidth
              label="Model 2 System Prompt"
              value={createForm.config.model2_config.system_prompt}
              onChange={(e) => setCreateForm({ 
                ...createForm, 
                config: { 
                  ...createForm.config, 
                  model2_config: { ...createForm.config.model2_config, system_prompt: e.target.value }
                } 
              })}
              rows={4}
              sx={{ mb: 2 }}
              placeholder="Enter system prompt for Model 2..."
            />
            <TextField
              fullWidth
              type="number"
              label="Max Turns"
              value={createForm.config.max_turns}
              onChange={(e) => setCreateForm({ ...createForm, config: { ...createForm.config, max_turns: parseInt(e.target.value) || 10 } })}
              inputProps={{ min: 5, max: 100 }}
              helperText="Maximum number of turns (5-100, default: 10)"
              sx={{ mb: 2 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setOpenCreateDialog(false); setOpenEditDialog(false); }}>Cancel</Button>
          <Button
            variant="contained"
            onClick={editingConfigId ? handleUpdate : handleCreate}
            disabled={(createMutation.isLoading || updateMutation.isLoading) || !createForm.name.trim() || !createForm.config.model1_config.system_prompt.trim() || !createForm.config.model2_config.system_prompt.trim()}
          >
            {editingConfigId ? (updateMutation.isLoading ? 'Updating...' : 'Update') : (createMutation.isLoading ? 'Creating...' : 'Create')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteConfirmDialog.open} onClose={() => setDeleteConfirmDialog({ open: false, configId: null })}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>Are you sure you want to delete this conversation configuration? This action cannot be undone.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmDialog({ open: false, configId: null })}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">Delete</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Conversation;
