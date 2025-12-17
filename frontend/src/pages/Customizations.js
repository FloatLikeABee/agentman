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
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';

const Customizations = () => {
  const queryClient = useQueryClient();
  const { data: profiles = [], isLoading, error } = useQuery('customizations', api.getCustomizations);

  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [createForm, setCreateForm] = useState({
    name: '',
    description: '',
    system_prompt: '',
    rag_collection: '',
  });
  const [queryText, setQueryText] = useState('');
  const [queryResult, setQueryResult] = useState('');
  const [queryMeta, setQueryMeta] = useState(null);
  const [queryError, setQueryError] = useState('');

  const createMutation = useMutation(api.createCustomization, {
    onSuccess: () => {
      queryClient.invalidateQueries('customizations');
      setOpenCreateDialog(false);
      setCreateForm({
        name: '',
        description: '',
        system_prompt: '',
        rag_collection: '',
      });
    },
  });

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
    });
  };

  const handleRunQuery = async () => {
    if (!selectedProfile || !queryText.trim()) return;
    try {
      setQueryError('');
      setQueryResult('');
      setQueryMeta(null);
      const res = await api.queryCustomization(selectedProfile.id, {
        query: queryText,
      });
      setQueryResult(res.response || '');
      setQueryMeta({
        model_used: res.model_used,
        rag_collection_used: res.rag_collection_used,
      });
    } catch (err) {
      console.error('Customization query error:', err);
      setQueryError('Failed to run customization. Please try again.');
    }
  };

  const handleDeleteProfile = (profileId) => {
    if (window.confirm('Are you sure you want to delete this customization?')) {
      deleteMutation.mutate(profileId);
    }
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
                        {profile.model_name && (
                          <Chip size="small" label={profile.model_name} variant="outlined" />
                        )}
                      </Box>
                    </Box>
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
                    onClick={handleRunQuery}
                    disabled={!queryText.trim()}
                  >
                    Run
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
                          border: '1px solid rgba(0,0,0,0.12)',
                          borderRadius: 1,
                          p: 2,
                          maxHeight: 260,
                          overflowY: 'auto',
                          bgColor: 'grey.50',
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
            <TextField
              fullWidth
              label="RAG Collection (optional)"
              value={createForm.rag_collection}
              onChange={(e) => setCreateForm({ ...createForm, rag_collection: e.target.value })}
              sx={{ mb: 1 }}
              helperText="Name of an existing RAG collection to use as context (optional)"
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
    </Box>
  );
};

export default Customizations;


