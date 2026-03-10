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
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  Edit as EditIcon,
  Psychology as AdviserIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import ReactMarkdown from 'react-markdown';
import api from '../services/api';

const AdviserManager = () => {
  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openRunDialog, setOpenRunDialog] = useState(false);
  const [selectedAdviser, setSelectedAdviser] = useState(null);
  const [editingAdviserId, setEditingAdviserId] = useState(null);
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState({ open: false, adviserId: null });
  const [queryText, setQueryText] = useState('');
  const [adviserResponse, setAdviserResponse] = useState('');
  const [fileInputs, setFileInputs] = useState([]);

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    draft_system_prompt: '',
    llm_provider: '',
    model_name: '',
    existing_rag_collections: [],
  });

  const queryClient = useQueryClient();
  const { data: advisers, isLoading } = useQuery('advisers', api.getAdvisers, {
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
  const { data: collections } = useQuery('collections', api.getRAGCollections, {
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
  const { data: models } = useQuery('models', api.getModels, {
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
  const { data: providersData } = useQuery('providers', api.getProviders, {
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      draft_system_prompt: '',
      llm_provider: '',
      model_name: '',
      existing_rag_collections: [],
    });
    setFileInputs([]);
  };

  const createAdviserMutation = useMutation(api.createAdviser, {
    onSuccess: () => {
      queryClient.invalidateQueries('advisers');
      setOpenCreateDialog(false);
      setEditingAdviserId(null);
      resetForm();
    },
  });

  const updateAdviserMutation = useMutation(
    ({ adviserId, payload }) => api.updateAdviser(adviserId, payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('advisers');
        setOpenCreateDialog(false);
        setEditingAdviserId(null);
        resetForm();
      },
    }
  );

  const deleteAdviserMutation = useMutation(api.deleteAdviser, {
    onSuccess: () => {
      queryClient.invalidateQueries('advisers');
      setDeleteConfirmDialog({ open: false, adviserId: null });
    },
  });

  const runAdviserMutation = useMutation(
    ({ adviserId, query, context }) => api.runAdviser(adviserId, query, context),
    {
      onSuccess: (data) => {
        setAdviserResponse(data.response);
      },
    }
  );

  useEffect(() => {
    if (openRunDialog && selectedAdviser) {
      setQueryText('');
      setAdviserResponse('');
    }
  }, [openRunDialog, selectedAdviser]);

  const buildPayloadFromForm = () => {
    return {
      name: formData.name,
      description: formData.description || null,
      draft_system_prompt: formData.draft_system_prompt,
      llm_provider: formData.llm_provider || null,
      model_name: formData.model_name || null,
      existing_rag_collections: formData.existing_rag_collections || [],
      files: fileInputs.map((f) => ({
        filename: f.filename,
        format: f.format,
        content: f.content,
        description: f.description || null,
      })),
    };
  };

  const handleCreateAdviser = () => {
    const payload = buildPayloadFromForm();
    if (editingAdviserId) {
      updateAdviserMutation.mutate({ adviserId: editingAdviserId, payload });
    } else {
      createAdviserMutation.mutate(payload);
    }
  };

  const handleEditAdviser = async (adviserId) => {
    try {
      const adviser = await api.getAdviser(adviserId);
      const config = adviser || {};

      setFormData({
        name: config.name || '',
        description: config.description || '',
        draft_system_prompt: config.system_prompt || '',
        llm_provider: config.llm_provider || '',
        model_name: config.model_name || '',
        existing_rag_collections: config.rag_collections || [],
      });
      setFileInputs([]);
      setEditingAdviserId(adviserId);
      setOpenCreateDialog(true);
    } catch (error) {
      console.error('Error loading adviser for editing:', error);
      alert('Failed to load adviser details. Please try again.');
    }
  };

  const handleDeleteAdviser = (adviserId) => {
    setDeleteConfirmDialog({ open: true, adviserId });
  };

  const handleDeleteConfirm = () => {
    deleteAdviserMutation.mutate(deleteConfirmDialog.adviserId);
  };

  const handleRunAdviser = () => {
    if (selectedAdviser && queryText) {
      runAdviserMutation.mutate({
        adviserId: selectedAdviser.id,
        query: queryText,
      });
    }
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;

    files.forEach((file) => {
      const fileName = file.name.toLowerCase();
      let detectedFormat = 'txt';
      if (fileName.endsWith('.json')) {
        detectedFormat = 'json';
      } else if (fileName.endsWith('.csv')) {
        detectedFormat = 'csv';
      } else if (fileName.endsWith('.txt')) {
        detectedFormat = 'txt';
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        setFileInputs((prev) => [
          ...prev,
          {
            filename: file.name,
            format: detectedFormat,
            content,
            description: '',
          },
        ]);
      };
      reader.readAsText(file);
    });

    // Allow selecting the same file again later
    event.target.value = '';
  };

  const handleFileDescriptionChange = (index, value) => {
    setFileInputs((prev) =>
      prev.map((f, i) => (i === index ? { ...f, description: value } : f))
    );
  };

  const handleRemoveFile = (index) => {
    setFileInputs((prev) => prev.filter((_, i) => i !== index));
  };

  const isInitialLoading = isLoading && !advisers;

  return (
    <Box>
      {isInitialLoading && <LinearProgress sx={{ mb: 2 }} />}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AdviserIcon />
          Adviser Manager
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            resetForm();
            setEditingAdviserId(null);
            setOpenCreateDialog(true);
          }}
        >
          Create Adviser
        </Button>
      </Box>

      <Grid container spacing={3}>
        {advisers?.map((adviser) => (
          <Grid item xs={12} md={6} lg={4} key={adviser.id}>
            <Card
              sx={{
                boxShadow: 2,
                transition: 'transform 0.2s',
                '&:hover': { transform: 'translateY(-2px)' },
                height: '100%',
              }}
            >
              <CardContent sx={{ p: 3, minHeight: 220, display: 'flex', flexDirection: 'column' }}>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    mb: 2,
                    flexGrow: 1,
                  }}
                >
                  <Box sx={{ flex: 1, pr: 1 }}>
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      {adviser.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {adviser.description}
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {adviser.model_name && (
                        <Chip label={adviser.model_name} size="small" variant="outlined" />
                      )}
                      {adviser.rag_collections && adviser.rag_collections.length > 0 && (
                        <Chip
                          label={`RAG: ${adviser.rag_collections.length}`}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      )}
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    <IconButton
                      size="small"
                      color="primary"
                      onClick={() => {
                        setQueryText('');
                        setAdviserResponse('');
                        setSelectedAdviser(adviser);
                        setOpenRunDialog(true);
                      }}
                      sx={{
                        bgcolor: 'primary.light',
                        '&:hover': { bgcolor: 'primary.main', color: 'white' },
                      }}
                      title="Run Adviser"
                    >
                      <RunIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="secondary"
                      onClick={() => handleEditAdviser(adviser.id)}
                      sx={{
                        bgcolor: 'secondary.light',
                        '&:hover': { bgcolor: 'secondary.main', color: 'white' },
                      }}
                      title="Edit Adviser"
                    >
                      <EditIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleDeleteAdviser(adviser.id)}
                      sx={{
                        bgcolor: 'error.light',
                        '&:hover': { bgcolor: 'error.main', color: 'white' },
                      }}
                      title="Delete Adviser"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Create/Edit Adviser Dialog */}
      <Dialog
        open={openCreateDialog}
        onClose={() => {
          setOpenCreateDialog(false);
          setEditingAdviserId(null);
          resetForm();
        }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ pb: 1 }}>
          {editingAdviserId ? 'Edit Adviser' : 'Create New Adviser'}
        </DialogTitle>
        <DialogContent sx={{ pt: 1 }}>
          {(createAdviserMutation.isError || updateAdviserMutation.isError) && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {editingAdviserId
                ? 'Failed to update adviser. Please check your inputs and try again.'
                : 'Failed to create adviser. Please check your inputs and try again.'}
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
                label="Adviser Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                required
                error={!formData.name.trim()}
                helperText={!formData.name.trim() ? 'Name is required' : ''}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Short Description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                multiline
                rows={2}
              />
            </Grid>

            {/* Model Settings */}
            <Grid item xs={12} sx={{ mt: 1 }}>
              <Typography variant="h6" gutterBottom color="primary">
                Model Settings
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>LLM Provider</InputLabel>
                <Select
                  value={formData.llm_provider}
                  label="LLM Provider"
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
                  label="Model"
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

            {/* Knowledge Sources */}
            <Grid item xs={12} sx={{ mt: 1 }}>
              <Typography variant="h6" gutterBottom color="primary">
                Knowledge Sources
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Existing RAG Collections</InputLabel>
                <Select
                  multiple
                  value={formData.existing_rag_collections}
                  label="Existing RAG Collections"
                  onChange={(e) =>
                    setFormData({ ...formData, existing_rag_collections: e.target.value })
                  }
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

            {/* File Inputs */}
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Base Data Files (JSON / CSV / TXT)
              </Typography>
              <Button variant="outlined" component="label">
                Upload Files
                <input
                  type="file"
                  hidden
                  multiple
                  accept=".json,.csv,.txt"
                  onChange={handleFileSelect}
                />
              </Button>
              <Box sx={{ mt: 1 }}>
                {fileInputs.map((file, index) => (
                  <Box
                    key={`${file.filename}-${index}`}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                      mt: 1,
                      p: 1,
                      borderRadius: 1,
                      border: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Chip label={`${file.filename} (${file.format})`} size="small" />
                    <TextField
                      fullWidth
                      size="small"
                      label="File Description (optional)"
                      value={file.description}
                      onChange={(e) => handleFileDescriptionChange(index, e.target.value)}
                    />
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleRemoveFile(index)}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Box>
                ))}
                {fileInputs.length === 0 && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    No files attached yet. You can create an adviser using only existing RAG
                    collections and web search, or attach JSON/CSV/TXT files as extra knowledge.
                  </Typography>
                )}
              </Box>
            </Grid>

            {/* System Prompt */}
            <Grid item xs={12} sx={{ mt: 1 }}>
              <Typography variant="h6" gutterBottom color="primary">
                Behaviour (System Prompt)
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Draft System Prompt"
                value={formData.draft_system_prompt}
                onChange={(e) =>
                  setFormData({ ...formData, draft_system_prompt: e.target.value })
                }
                multiline
                rows={4}
                required
                error={!formData.draft_system_prompt.trim()}
                helperText={
                  !formData.draft_system_prompt.trim()
                    ? 'System prompt is required. It will be cleaned up by the AI model when saving.'
                    : 'Describe how this adviser should behave. The AI will refine this draft into a final system prompt.'
                }
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions sx={{ p: 3, pt: 1 }}>
          <Button
            onClick={() => {
              setOpenCreateDialog(false);
              setEditingAdviserId(null);
              resetForm();
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleCreateAdviser}
            variant="contained"
            disabled={
              createAdviserMutation.isLoading ||
              updateAdviserMutation.isLoading ||
              !formData.name.trim() ||
              !formData.draft_system_prompt.trim()
            }
          >
            {editingAdviserId
              ? updateAdviserMutation.isLoading
                ? 'Updating...'
                : 'Update Adviser'
              : createAdviserMutation.isLoading
              ? 'Creating...'
              : 'Create Adviser'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Run Adviser Dialog */}
      <Dialog
        open={openRunDialog}
        onClose={() => {
          setOpenRunDialog(false);
          setQueryText('');
          setAdviserResponse('');
        }}
        maxWidth="lg"
        fullWidth
        sx={{
          '& .MuiDialog-paper': {
            height: '80vh',
            maxHeight: '600px',
          },
        }}
      >
        <DialogTitle sx={{ pb: 1 }}>Run Adviser: {selectedAdviser?.name}</DialogTitle>
        <DialogContent sx={{ pt: 1, display: 'flex', flexDirection: 'column' }}>
          {runAdviserMutation.isError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to run adviser. Please try again.
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
              placeholder="Enter your query for the adviser..."
              sx={{ mb: 2 }}
            />
            <Button
              variant="contained"
              onClick={handleRunAdviser}
              disabled={runAdviserMutation.isLoading || !queryText.trim()}
              size="large"
              fullWidth
            >
              {runAdviserMutation.isLoading ? 'Running Adviser...' : 'Run Adviser'}
            </Button>
          </Box>

          {/* Response Section */}
          {adviserResponse && (
            <Box sx={{ flex: 1, minHeight: 0 }}>
              <Typography
                variant="h6"
                gutterBottom
                sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
              >
                Response
              </Typography>
              <Box
                sx={{
                  mt: 1,
                  p: 2,
                  bgcolor: 'background.paper',
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'primary.main',
                  borderOpacity: 0.3,
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
                      bgcolor: 'background.default',
                      borderRadius: '4px',
                    },
                    '&::-webkit-scrollbar-thumb': {
                      bgcolor: 'primary.main',
                      bgcolorOpacity: 0.5,
                      borderRadius: '4px',
                      '&:hover': {
                        bgcolor: 'primary.light',
                      },
                    },
                  }}
                >
                  <Box sx={{ lineHeight: 1.6 }}>
                    <ReactMarkdown>{adviserResponse}</ReactMarkdown>
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
        onClose={() => setDeleteConfirmDialog({ open: false, adviserId: null })}
      >
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this adviser? The underlying agent will be removed,
            but any RAG collections created from files will be preserved.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmDialog({ open: false, adviserId: null })}>
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

export default AdviserManager;

