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
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  Edit as EditIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';

const Customizations = () => {
  const queryClient = useQueryClient();
  const { data: profiles = [], isLoading, error } = useQuery('customizations', api.getCustomizations, { staleTime: 5 * 60 * 1000 }); // Cache for 5 minutes
  const { data: models = [] } = useQuery('models', api.getModels, { enabled: true, staleTime: 5 * 60 * 1000 });
  const { data: providersData } = useQuery('providers', api.getProviders, { enabled: true, staleTime: 5 * 60 * 1000 });
  const { data: collections = [] } = useQuery('collections', api.getRAGCollections, { staleTime: 5 * 60 * 1000 });

  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [editingProfileId, setEditingProfileId] = useState(null);
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState({ open: false, profileId: null });
  const [createForm, setCreateForm] = useState({
    name: '',
    description: '',
    system_prompt: '',
    rag_collection: '',
    llm_provider: '',
    model_name: '',
  });
  const [queryText, setQueryText] = useState('');
  const [queryResult, setQueryResult] = useState('');
  const [queryMeta, setQueryMeta] = useState(null);
  const [queryError, setQueryError] = useState('');
  const [isRunningQuery, setIsRunningQuery] = useState(false);

  const createMutation = useMutation(api.createCustomization, {
    onSuccess: () => {
      queryClient.invalidateQueries('customizations');
      setOpenCreateDialog(false);
      setCreateForm({
        name: '',
        description: '',
        system_prompt: '',
        rag_collection: '',
        llm_provider: '',
        model_name: '',
      });
    },
  });

  const updateMutation = useMutation(
    ({ profileId, payload }) => api.updateCustomization(profileId, payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('customizations');
        setOpenEditDialog(false);
        setEditingProfileId(null);
        setCreateForm({
          name: '',
          description: '',
          system_prompt: '',
          rag_collection: '',
          llm_provider: '',
          model_name: '',
        });
      },
    }
  );

  const deleteMutation = useMutation(api.deleteCustomization, {
    onSuccess: () => {
      queryClient.invalidateQueries('customizations');
      if (selectedProfile) {
        setSelectedProfile(null);
        setQueryText('');
        setQueryResult('');
        setQueryMeta(null);
      }
    },
  });

  const handleCreate = () => {
    createMutation.mutate({
      name: createForm.name,
      description: createForm.description,
      system_prompt: createForm.system_prompt,
      rag_collection: createForm.rag_collection || null,
      llm_provider: createForm.llm_provider || null,
      model_name: createForm.model_name || null,
    });
  };

  const handleRunQuery = async (e) => {
    e?.preventDefault?.();
    e?.stopPropagation?.();
    
    if (!selectedProfile || !queryText.trim()) {
      console.log('Cannot run query: no profile selected or query is empty', { selectedProfile, queryText });
      return;
    }
    
    console.log('Running customization query:', { profileId: selectedProfile.id, query: queryText });
    
    setIsRunningQuery(true);
    setQueryError('');
    setQueryResult('');
    setQueryMeta(null);
    
    try {
      const res = await api.queryCustomization(selectedProfile.id, {
        query: queryText,
      });
      
      console.log('Customization query response:', res);
      
      setQueryResult(res.response || '');
      setQueryMeta({
        model_used: res.model_used,
        rag_collection_used: res.rag_collection_used,
      });
    } catch (err) {
      console.error('Customization query error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to run customization. Please try again.';
      setQueryError(errorMessage);
      console.error('Full error object:', err);
    } finally {
      setIsRunningQuery(false);
    }
  };

  const handleEditProfile = (profile) => {
    setEditingProfileId(profile.id);
    setCreateForm({
      name: profile.name || '',
      description: profile.description || '',
      system_prompt: profile.system_prompt || '',
      rag_collection: profile.rag_collection || '',
      llm_provider: profile.llm_provider || '',
      model_name: profile.model_name || '',
    });
    setOpenEditDialog(true);
  };

  const handleUpdate = () => {
    updateMutation.mutate({
      profileId: editingProfileId,
      payload: {
        name: createForm.name,
        description: createForm.description,
        system_prompt: createForm.system_prompt,
        rag_collection: createForm.rag_collection || null,
        llm_provider: createForm.llm_provider || null,
        model_name: createForm.model_name || null,
      },
    });
  };

  const handleDeleteProfile = (profileId) => {
    setDeleteConfirmDialog({ open: true, profileId });
  };

  const handleDeleteConfirm = () => {
    deleteMutation.mutate(deleteConfirmDialog.profileId);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Customizations</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpenCreateDialog(true)}
        >
          New Customization
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load customizations
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Grid container spacing={2}>
            {profiles.map((profile) => (
              <Grid item xs={12} key={profile.id}>
                <Card
                  sx={{
                    boxShadow: 2,
                    cursor: 'pointer',
                    border:
                      selectedProfile && selectedProfile.id === profile.id
                        ? '2px solid #1976d2'
                        : '1px solid rgba(0,0,0,0.12)',
                  }}
                  onClick={() => setSelectedProfile(profile)}
                >
                  <CardContent sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Box sx={{ flex: 1, pr: 1 }}>
                      <Typography variant="h6">{profile.name}</Typography>
                      {profile.description && (
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                          {profile.description}
                        </Typography>
                      )}
                      <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        {profile.rag_collection && (
                          <Chip size="small" label={`RAG: ${profile.rag_collection}`} color="primary" variant="outlined" />
                        )}
                        {profile.llm_provider && (
                          <Chip size="small" label={`Provider: ${profile.llm_provider}`} variant="outlined" />
                        )}
                        {profile.model_name && (
                          <Chip size="small" label={`Model: ${profile.model_name}`} variant="outlined" />
                        )}
                      </Box>
                    </Box>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEditProfile(profile);
                        }}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        color="error"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteProfile(profile.id);
                        }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
            {!isLoading && profiles.length === 0 && (
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  No customizations yet. Create one to get started.
                </Typography>
              </Grid>
            )}
          </Grid>
        </Grid>

        {/* Playground */}
        <Grid item xs={12} md={6}>
          <Card sx={{ boxShadow: 2, height: '100%' }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Playground
              </Typography>
              {selectedProfile ? (
                <>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'medium', mb: 1 }}>
                    {selectedProfile.name}
                  </Typography>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{
                      mb: 2,
                      whiteSpace: 'pre-wrap',
                      maxHeight: 120,
                      overflowY: 'auto',
                      border: '1px solid rgba(0,0,0,0.12)',
                      borderRadius: 1,
                      p: 1,
                    }}
                  >
                    {selectedProfile.system_prompt}
                  </Typography>

                  <TextField
                    fullWidth
                    label="Short User Prompt"
                    value={queryText}
                    onChange={(e) => setQueryText(e.target.value)}
                    multiline
                    rows={3}
                    sx={{ mb: 2 }}
                  />
                  <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    fullWidth
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      handleRunQuery(e);
                    }}
                    disabled={!queryText.trim() || !selectedProfile || isRunningQuery}
                    type="button"
                  >
                    {isRunningQuery ? 'Running...' : 'Run'}
                  </Button>

                  {queryError && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      {queryError}
                    </Alert>
                  )}

                  {queryResult && (
                    <Box sx={{ mt: 3 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Response
                      </Typography>
                      <Box
                        sx={{
                          whiteSpace: 'pre-wrap',
                          border: '1px solid',
                          borderColor: 'primary.main',
                          borderOpacity: 0.3,
                          borderRadius: 1,
                          p: 2,
                          maxHeight: 260,
                          overflowY: 'auto',
                          bgcolor: 'background.paper',
                          color: 'text.primary',
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
                        {queryResult}
                      </Box>
                      {queryMeta && (
                        <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                          {queryMeta.model_used && (
                            <Chip size="small" label={`Model: ${queryMeta.model_used}`} />
                          )}
                          {queryMeta.rag_collection_used && (
                            <Chip size="small" label={`RAG: ${queryMeta.rag_collection_used}`} />
                          )}
                        </Box>
                      )}
                    </Box>
                  )}
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Select a customization from the list to start testing it with short prompts.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Edit Customization Dialog */}
      <Dialog
        open={openEditDialog}
        onClose={() => {
          setOpenEditDialog(false);
          setEditingProfileId(null);
          setCreateForm({
            name: '',
            description: '',
            system_prompt: '',
            rag_collection: '',
            llm_provider: '',
            model_name: '',
          });
        }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Edit Customization</DialogTitle>
        <DialogContent>
          {updateMutation.isError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to update customization. Please check your inputs and try again.
            </Alert>
          )}
          <Box sx={{ mt: 1 }}>
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
            <TextField
              fullWidth
              label="System Prompt / Instructions"
              value={createForm.system_prompt}
              onChange={(e) => setCreateForm({ ...createForm, system_prompt: e.target.value })}
              multiline
              rows={6}
              sx={{ mb: 2 }}
              placeholder="Describe how the AI should behave for this customization..."
            />
            <Autocomplete
              freeSolo
              options={collections.map((col) => col.name)}
              value={createForm.rag_collection}
              onChange={(event, newValue) => {
                setCreateForm({ ...createForm, rag_collection: newValue || '' });
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="RAG Collection (optional)"
                  helperText="Name of an existing RAG collection to use as context (optional)"
                  sx={{ mb: 2 }}
                />
              )}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>LLM Provider (optional)</InputLabel>
              <Select
                value={createForm.llm_provider}
                label="LLM Provider (optional)"
                onChange={(e) => setCreateForm({ ...createForm, llm_provider: e.target.value })}
              >
                <MenuItem value="">
                  <em>Use default provider</em>
                </MenuItem>
                {providersData?.providers?.map((provider) => (
                  <MenuItem key={provider} value={provider}>
                    {provider}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Autocomplete
              freeSolo
              options={models.map((model) => model.name)}
              value={createForm.model_name}
              onChange={(event, newValue) => {
                setCreateForm({ ...createForm, model_name: newValue || '' });
              }}
              onInputChange={(event, newInputValue) => {
                setCreateForm({ ...createForm, model_name: newInputValue });
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Model Name (optional)"
                  helperText="Choose from available models or enter any model name you want"
                  placeholder="e.g., gemini-2.5-flash, qwen3-max, or any custom model"
                />
              )}
              sx={{ mb: 1 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenEditDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleUpdate}
            disabled={updateMutation.isLoading || !createForm.name.trim() || !createForm.system_prompt.trim()}
          >
            {updateMutation.isLoading ? 'Updating...' : 'Update'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create Customization Dialog */}
      <Dialog open={openCreateDialog} onClose={() => setOpenCreateDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Customization</DialogTitle>
        <DialogContent>
          {createMutation.isError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to create customization. Please check your inputs and try again.
            </Alert>
          )}
          <Box sx={{ mt: 1 }}>
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
            <TextField
              fullWidth
              label="System Prompt / Instructions"
              value={createForm.system_prompt}
              onChange={(e) => setCreateForm({ ...createForm, system_prompt: e.target.value })}
              multiline
              rows={6}
              sx={{ mb: 2 }}
              placeholder="Describe how the AI should behave for this customization..."
            />
            <Autocomplete
              freeSolo
              options={collections.map((col) => col.name)}
              value={createForm.rag_collection}
              onChange={(event, newValue) => {
                setCreateForm({ ...createForm, rag_collection: newValue || '' });
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="RAG Collection (optional)"
                  helperText="Name of an existing RAG collection to use as context (optional)"
                  sx={{ mb: 2 }}
                />
              )}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>LLM Provider (optional)</InputLabel>
              <Select
                value={createForm.llm_provider}
                label="LLM Provider (optional)"
                onChange={(e) => setCreateForm({ ...createForm, llm_provider: e.target.value })}
              >
                <MenuItem value="">
                  <em>Use default provider</em>
                </MenuItem>
                {providersData?.providers?.map((provider) => (
                  <MenuItem key={provider} value={provider}>
                    {provider}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Autocomplete
              freeSolo
              options={models.map((model) => model.name)}
              value={createForm.model_name}
              onChange={(event, newValue) => {
                setCreateForm({ ...createForm, model_name: newValue || '' });
              }}
              onInputChange={(event, newInputValue) => {
                setCreateForm({ ...createForm, model_name: newInputValue });
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Model Name (optional)"
                  helperText="Choose from available models or enter any model name you want"
                  placeholder="e.g., gemini-2.5-flash, qwen3-max, or any custom model"
                />
              )}
              sx={{ mb: 1 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenCreateDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleCreate}
            disabled={createMutation.isLoading || !createForm.name.trim() || !createForm.system_prompt.trim()}
          >
            {createMutation.isLoading ? 'Creating...' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmDialog.open}
        onClose={() => setDeleteConfirmDialog({ open: false, profileId: null })}
      >
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this customization? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmDialog({ open: false, profileId: null })}>
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

export default Customizations;


