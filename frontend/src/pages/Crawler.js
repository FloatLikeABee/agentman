import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  FormControlLabel,
  Switch,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  LinearProgress,
  Grid,
  Chip,
  Divider,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Paper,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Send as SendIcon,
  Http as HttpIcon,
  Storage as StorageIcon,
} from '@mui/icons-material';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import api from '../services/api';

const Crawler = () => {
  const [url, setUrl] = useState('');
  const [useJs, setUseJs] = useState(false);
  const [llmProvider, setLlmProvider] = useState('');
  const [model, setModel] = useState('');
  const [collectionName, setCollectionName] = useState('');
  const [collectionDescription, setCollectionDescription] = useState('');

  const crawlMutation = useMutation(
    (data) => api.crawlWebsite(data),
    {
      onSuccess: (data) => {
        // Auto-fill collection name/description if AI generated them
        if (data.collection_name && !collectionName) {
          setCollectionName(data.collection_name);
        }
        if (data.collection_description && !collectionDescription) {
          setCollectionDescription(data.collection_description);
        }
      },
    }
  );

  const handleCrawl = () => {
    if (!url.trim()) {
      return;
    }

    crawlMutation.mutate({
      url: url.trim(),
      use_js: useJs,
      llm_provider: llmProvider || null,
      model: model || null,
      collection_name: collectionName || null,
      collection_description: collectionDescription || null,
    });
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Web Crawler</Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Crawler Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                Crawl Configuration
              </Typography>

              {crawlMutation.isError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {crawlMutation.error?.response?.data?.detail || crawlMutation.error?.message || 'Failed to crawl website. Please try again.'}
                </Alert>
              )}

              {crawlMutation.isSuccess && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  Successfully crawled and saved to RAG collection!
                </Alert>
              )}

              <TextField
                fullWidth
                label="Website URL"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                sx={{ mb: 2 }}
                required
                error={!url.trim()}
                helperText={!url.trim() ? 'URL is required' : 'Enter the website URL to crawl'}
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={useJs}
                    onChange={(e) => setUseJs(e.target.checked)}
                  />
                }
                label="Use JavaScript Rendering (for dynamic content)"
                sx={{ mb: 2, display: 'block' }}
              />

              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>LLM Provider (Optional)</InputLabel>
                    <Select
                      value={llmProvider}
                      onChange={(e) => setLlmProvider(e.target.value)}
                      label="LLM Provider (Optional)"
                    >
                      <MenuItem value="">Auto (Use Default)</MenuItem>
                      <MenuItem value="gemini">Gemini</MenuItem>
                      <MenuItem value="qwen">Qwen</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Model (Optional)"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder="Leave empty for default"
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Collection Settings (Optional - AI will generate if left empty)
              </Typography>

              <TextField
                fullWidth
                label="Collection Name (Optional)"
                value={collectionName}
                onChange={(e) => setCollectionName(e.target.value)}
                placeholder="AI will generate a name if left empty"
                sx={{ mb: 2 }}
                helperText="Leave empty to let AI generate a collection name"
              />

              <TextField
                fullWidth
                label="Collection Description (Optional)"
                value={collectionDescription}
                onChange={(e) => setCollectionDescription(e.target.value)}
                placeholder="AI will generate a description if left empty"
                multiline
                rows={2}
                sx={{ mb: 2 }}
                helperText="Leave empty to let AI generate a description"
              />

              <Button
                variant="contained"
                size="large"
                startIcon={<PlayIcon />}
                onClick={handleCrawl}
                disabled={crawlMutation.isLoading || !url.trim()}
                fullWidth
              >
                {crawlMutation.isLoading ? 'Crawling...' : 'Start Crawling'}
              </Button>

              {crawlMutation.isLoading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
                    This may take 1-2 minutes. Please wait...
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Request Tools Section */}
        <Grid item xs={12} md={6}>
          <RequestToolsPanel />
        </Grid>
      </Grid>
    </Box>
  );
};

// Request Tools Panel Component
const RequestToolsPanel = () => {
  const queryClient = useQueryClient();
  const { data: requests = [], isLoading } = useQuery('request-tools', api.getRequestTools);
  const [selectedRequest, setSelectedRequest] = useState(null);
  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [editingRequestId, setEditingRequestId] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [createForm, setCreateForm] = useState({
    name: '',
    description: '',
    request_type: 'http',
    method: 'GET',
    url: '',
    endpoint: '',
    headers: {},
    params: {},
    body: '',
    timeout: 30,
  });
  const [headersText, setHeadersText] = useState('');
  const [paramsText, setParamsText] = useState('');

  const createMutation = useMutation(api.createRequestTool, {
    onSuccess: () => {
      queryClient.invalidateQueries('request-tools');
      setOpenCreateDialog(false);
      resetForm();
    },
  });

  const updateMutation = useMutation(
    ({ requestId, payload }) => api.updateRequestTool(requestId, payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('request-tools');
        setOpenEditDialog(false);
        setEditingRequestId(null);
        resetForm();
      },
    }
  );

  const deleteMutation = useMutation(api.deleteRequestTool, {
    onSuccess: () => {
      queryClient.invalidateQueries('request-tools');
      if (selectedRequest) {
        setSelectedRequest(null);
      }
    },
  });

  const executeMutation = useMutation(api.executeRequestTool, {
    onSuccess: () => {
      queryClient.invalidateQueries('request-tools');
    },
  });

  const resetForm = () => {
    setCreateForm({
      name: '',
      description: '',
      request_type: 'http',
      method: 'GET',
      url: '',
      endpoint: '',
      headers: {},
      params: {},
      body: '',
      timeout: 30,
    });
    setHeadersText('');
    setParamsText('');
  };

  const handleCreate = () => {
    const payload = {
      ...createForm,
      headers: parseJsonOrEmpty(headersText),
      params: parseJsonOrEmpty(paramsText),
      body: createForm.body ? (tryParseJson(createForm.body) || createForm.body) : null,
    };
    createMutation.mutate(payload);
  };

  const handleEdit = (request) => {
    setEditingRequestId(request.id);
    setCreateForm({
      name: request.name,
      description: request.description || '',
      request_type: request.request_type,
      method: request.method || 'GET',
      url: request.url || '',
      endpoint: request.endpoint || '',
      headers: request.headers || {},
      params: request.params || {},
      body: typeof request.body === 'string' ? request.body : JSON.stringify(request.body || {}, null, 2),
      timeout: request.timeout || 30,
    });
    setHeadersText(JSON.stringify(request.headers || {}, null, 2));
    setParamsText(JSON.stringify(request.params || {}, null, 2));
    setOpenEditDialog(true);
  };

  const handleUpdate = () => {
    const payload = {
      ...createForm,
      headers: parseJsonOrEmpty(headersText),
      params: parseJsonOrEmpty(paramsText),
      body: createForm.body ? (tryParseJson(createForm.body) || createForm.body) : null,
    };
    updateMutation.mutate({ requestId: editingRequestId, payload });
  };

  const handleExecute = (requestId) => {
    executeMutation.mutate(requestId);
  };

  const parseJsonOrEmpty = (text) => {
    if (!text || !text.trim()) return {};
    try {
      return JSON.parse(text);
    } catch {
      return {};
    }
  };

  const tryParseJson = (text) => {
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" color="primary">
            Request Tools
          </Typography>
          <Button
            size="small"
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => {
              resetForm();
              setOpenCreateDialog(true);
            }}
          >
            New Request
          </Button>
        </Box>

        <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} sx={{ mb: 2 }}>
          <Tab label="Requests" icon={<HttpIcon />} iconPosition="start" />
          <Tab label="Response" icon={<StorageIcon />} iconPosition="start" />
        </Tabs>

        {tabValue === 0 && (
          <Box>
            {requests.length === 0 ? (
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                No requests configured. Create one to get started.
              </Typography>
            ) : (
              <Box sx={{ maxHeight: 400, overflowY: 'auto' }}>
                {requests.map((req) => (
                  <Card
                    key={req.id}
                    sx={{
                      mb: 1,
                      cursor: 'pointer',
                      border: selectedRequest?.id === req.id ? '2px solid #1976d2' : '1px solid rgba(0,0,0,0.12)',
                    }}
                    onClick={() => setSelectedRequest(req)}
                  >
                    <CardContent sx={{ p: 1.5 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="subtitle1">{req.name}</Typography>
                          <Chip
                            size="small"
                            label={req.request_type.toUpperCase()}
                            color={req.request_type === 'http' ? 'primary' : 'secondary'}
                            sx={{ mt: 0.5, mr: 0.5 }}
                          />
                          {req.method && (
                            <Chip size="small" label={req.method} variant="outlined" sx={{ mt: 0.5, mr: 0.5 }} />
                          )}
                          {req.last_executed_at && (
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                              Last: {new Date(req.last_executed_at).toLocaleString()}
                            </Typography>
                          )}
                        </Box>
                        <Box>
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleExecute(req.id);
                            }}
                          >
                            <SendIcon fontSize="small" />
                          </IconButton>
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEdit(req);
                            }}
                          >
                            <EditIcon fontSize="small" />
                          </IconButton>
                          <IconButton
                            size="small"
                            color="error"
                            onClick={(e) => {
                              e.stopPropagation();
                              if (window.confirm('Delete this request?')) {
                                deleteMutation.mutate(req.id);
                              }
                            }}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                ))}
              </Box>
            )}
          </Box>
        )}

        {tabValue === 1 && selectedRequest && (
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              {selectedRequest.name}
            </Typography>
            {selectedRequest.last_response ? (
              <Box>
                <Chip
                  label={selectedRequest.last_response.success ? 'Success' : 'Failed'}
                  color={selectedRequest.last_response.success ? 'success' : 'error'}
                  sx={{ mb: 1 }}
                />
                {selectedRequest.last_response.status_code && (
                  <Chip
                    label={`Status: ${selectedRequest.last_response.status_code}`}
                    sx={{ mb: 1, ml: 1 }}
                  />
                )}
                <Paper sx={{ p: 2, mt: 2, maxHeight: 300, overflow: 'auto' }}>
                  <Typography variant="caption" color="text.secondary">
                    Response Data:
                  </Typography>
                  <pre style={{ margin: 0, fontSize: '0.875rem' }}>
                    {JSON.stringify(selectedRequest.last_response.response_data, null, 2)}
                  </pre>
                </Paper>
                {selectedRequest.last_response.error && (
                  <Alert severity="error" sx={{ mt: 1 }}>
                    {selectedRequest.last_response.error}
                  </Alert>
                )}
              </Box>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No response yet. Execute the request to see results.
              </Typography>
            )}
          </Box>
        )}

        {executeMutation.isLoading && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
              Executing request...
            </Typography>
          </Box>
        )}

        {executeMutation.isError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {executeMutation.error?.response?.data?.detail || executeMutation.error?.message || 'Execution failed'}
          </Alert>
        )}

        {executeMutation.isSuccess && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Request executed successfully!
          </Alert>
        )}

        {/* Create Dialog */}
        <Dialog open={openCreateDialog} onClose={() => setOpenCreateDialog(false)} maxWidth="md" fullWidth>
          <DialogTitle>Create Request Configuration</DialogTitle>
          <DialogContent>
            <RequestForm
              form={createForm}
              setForm={setCreateForm}
              headersText={headersText}
              setHeadersText={setHeadersText}
              paramsText={paramsText}
              setParamsText={setParamsText}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setOpenCreateDialog(false)}>Cancel</Button>
            <Button
              variant="contained"
              onClick={handleCreate}
              disabled={createMutation.isLoading || !createForm.name.trim()}
            >
              {createMutation.isLoading ? 'Creating...' : 'Create'}
            </Button>
          </DialogActions>
        </Dialog>

        {/* Edit Dialog */}
        <Dialog open={openEditDialog} onClose={() => setOpenEditDialog(false)} maxWidth="md" fullWidth>
          <DialogTitle>Edit Request Configuration</DialogTitle>
          <DialogContent>
            <RequestForm
              form={createForm}
              setForm={setCreateForm}
              headersText={headersText}
              setHeadersText={setHeadersText}
              paramsText={paramsText}
              setParamsText={setParamsText}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setOpenEditDialog(false)}>Cancel</Button>
            <Button
              variant="contained"
              onClick={handleUpdate}
              disabled={updateMutation.isLoading || !createForm.name.trim()}
            >
              {updateMutation.isLoading ? 'Updating...' : 'Update'}
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  );
};

// Request Form Component
const RequestForm = ({ form, setForm, headersText, setHeadersText, paramsText, setParamsText }) => {
  return (
    <Box sx={{ mt: 1 }}>
      <TextField
        fullWidth
        label="Request Name (Unique)"
        value={form.name}
        onChange={(e) => setForm({ ...form, name: e.target.value })}
        sx={{ mb: 2 }}
        required
      />
      <TextField
        fullWidth
        label="Description"
        value={form.description}
        onChange={(e) => setForm({ ...form, description: e.target.value })}
        multiline
        rows={2}
        sx={{ mb: 2 }}
      />
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Request Type</InputLabel>
        <Select
          value={form.request_type}
          label="Request Type"
          onChange={(e) => setForm({ ...form, request_type: e.target.value })}
        >
          <MenuItem value="http">HTTP API</MenuItem>
          <MenuItem value="internal">Internal Service</MenuItem>
        </Select>
      </FormControl>
      {form.request_type === 'http' ? (
        <>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>HTTP Method</InputLabel>
            <Select
              value={form.method}
              label="HTTP Method"
              onChange={(e) => setForm({ ...form, method: e.target.value })}
            >
              <MenuItem value="GET">GET</MenuItem>
              <MenuItem value="POST">POST</MenuItem>
              <MenuItem value="PUT">PUT</MenuItem>
              <MenuItem value="DELETE">DELETE</MenuItem>
              <MenuItem value="PATCH">PATCH</MenuItem>
              <MenuItem value="HEAD">HEAD</MenuItem>
              <MenuItem value="OPTIONS">OPTIONS</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="URL"
            value={form.url}
            onChange={(e) => setForm({ ...form, url: e.target.value })}
            sx={{ mb: 2 }}
            required
            placeholder="https://api.example.com/endpoint"
          />
        </>
      ) : (
        <TextField
          fullWidth
          label="Internal Endpoint"
          value={form.endpoint}
          onChange={(e) => setForm({ ...form, endpoint: e.target.value })}
          sx={{ mb: 2 }}
          required
          placeholder="/api/endpoint"
        />
      )}
      <TextField
        fullWidth
        label="Headers (JSON)"
        value={headersText}
        onChange={(e) => setHeadersText(e.target.value)}
        multiline
        rows={3}
        sx={{ mb: 2 }}
        placeholder='{"Content-Type": "application/json", "Authorization": "Bearer token"}'
        helperText="JSON object with key-value pairs"
      />
      <TextField
        fullWidth
        label="Query Parameters (JSON)"
        value={paramsText}
        onChange={(e) => setParamsText(e.target.value)}
        multiline
        rows={3}
        sx={{ mb: 2 }}
        placeholder='{"param1": "value1", "param2": "value2"}'
        helperText="JSON object with key-value pairs"
      />
      <TextField
        fullWidth
        label="Body (JSON or Text)"
        value={form.body}
        onChange={(e) => setForm({ ...form, body: e.target.value })}
        multiline
        rows={4}
        sx={{ mb: 2 }}
        placeholder='{"key": "value"}'
      />
      <TextField
        fullWidth
        label="Timeout (seconds)"
        type="number"
        value={form.timeout}
        onChange={(e) => setForm({ ...form, timeout: parseFloat(e.target.value) || 30 })}
        inputProps={{ min: 1, max: 300 }}
        sx={{ mb: 2 }}
      />
    </Box>
  );
};

export default Crawler;

