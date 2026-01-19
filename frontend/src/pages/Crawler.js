import React, { useState, useMemo } from 'react';
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
  Paper,
  InputAdornment,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Send as SendIcon,
  Http as HttpIcon,
  Language as WebIcon,
  Search as SearchIcon,
  Api as ApiIcon,
} from '@mui/icons-material';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import api from '../services/api';

const Crawler = () => {
  const [activeTab, setActiveTab] = useState('crawlers'); // 'crawlers' or 'requests'

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3 }}>Data Sources</Typography>
      
      <Grid container spacing={3}>
        {/* Web Crawlers Tab */}
        <Grid item xs={12} md={6}>
          <Paper 
            sx={{ 
              p: 2, 
              height: '100%',
              border: '2px solid',
              borderColor: activeTab === 'crawlers' ? 'primary.main' : 'transparent',
              transition: 'border-color 0.2s',
            }}
            onClick={() => setActiveTab('crawlers')}
          >
            <WebCrawlersPanel isActive={activeTab === 'crawlers'} />
          </Paper>
        </Grid>

        {/* REST API Requests Tab */}
        <Grid item xs={12} md={6}>
          <Paper 
            sx={{ 
              p: 2, 
              height: '100%',
              border: '2px solid',
              borderColor: activeTab === 'requests' ? 'primary.main' : 'transparent',
              transition: 'border-color 0.2s',
            }}
            onClick={() => setActiveTab('requests')}
          >
            <RequestToolsPanel isActive={activeTab === 'requests'} />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

// Web Crawlers Panel Component
const WebCrawlersPanel = ({ isActive }) => {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState('');
  const [openProfileDialog, setOpenProfileDialog] = useState(false);
  const [editingProfileId, setEditingProfileId] = useState(null);
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState({ open: false, profileId: null });
  const [headersError, setHeadersError] = useState('');
  const [profileForm, setProfileForm] = useState({
    name: '',
    description: '',
    url: '',
    use_js: false,
    llm_provider: '',
    model: '',
    collection_name: '',
    collection_description: '',
    follow_links: false,
    max_depth: 3,
    max_pages: 50,
    same_domain_only: true,
    headers: '',
  });

  const { data: profiles = [], isLoading } = useQuery('crawler-profiles', api.getCrawlerProfiles, { staleTime: 5 * 60 * 1000 });

  const filteredProfiles = useMemo(() => {
    if (!searchQuery.trim()) return profiles;
    const query = searchQuery.toLowerCase();
    return profiles.filter(p => 
      p.name?.toLowerCase().includes(query) || 
      p.description?.toLowerCase().includes(query) ||
      p.url?.toLowerCase().includes(query)
    );
  }, [profiles, searchQuery]);

  const createProfileMutation = useMutation(api.createCrawlerProfile, {
    onSuccess: () => {
      queryClient.invalidateQueries('crawler-profiles');
      setOpenProfileDialog(false);
      resetProfileForm();
    },
  });

  const updateProfileMutation = useMutation(
    ({ profileId, payload }) => api.updateCrawlerProfile(profileId, payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('crawler-profiles');
        setOpenProfileDialog(false);
        setEditingProfileId(null);
        resetProfileForm();
      },
    }
  );

  const deleteProfileMutation = useMutation(api.deleteCrawlerProfile, {
    onSuccess: () => {
      queryClient.invalidateQueries('crawler-profiles');
      setDeleteConfirmDialog({ open: false, profileId: null });
    },
  });

  const executeProfileMutation = useMutation(api.executeCrawlerProfile, {
    onSuccess: () => {
      queryClient.invalidateQueries('collections');
      queryClient.invalidateQueries('crawler-profiles');
    },
  });

  const resetProfileForm = () => {
    setProfileForm({
      name: '',
      description: '',
      url: '',
      use_js: false,
      llm_provider: '',
      model: '',
      collection_name: '',
      collection_description: '',
      follow_links: false,
      max_depth: 3,
      max_pages: 50,
      same_domain_only: true,
      headers: '',
    });
    setHeadersError('');
  };

  const handleCreateProfile = () => {
    try {
      setHeadersError('');
      const headers = profileForm.headers.trim() ? JSON.parse(profileForm.headers) : null;
      createProfileMutation.mutate({
        ...profileForm,
        headers: headers,
        llm_provider: profileForm.llm_provider || null,
        model: profileForm.model || null,
        collection_name: profileForm.collection_name || null,
        collection_description: profileForm.collection_description || null,
      });
    } catch (e) {
      setHeadersError(`Invalid JSON: ${e.message}`);
    }
  };

  const handleUpdateProfile = () => {
    try {
      setHeadersError('');
      const headers = profileForm.headers.trim() ? JSON.parse(profileForm.headers) : null;
      updateProfileMutation.mutate({
        profileId: editingProfileId,
        payload: {
          ...profileForm,
          headers: headers,
          llm_provider: profileForm.llm_provider || null,
          model: profileForm.model || null,
          collection_name: profileForm.collection_name || null,
          collection_description: profileForm.collection_description || null,
        },
      });
    } catch (e) {
      setHeadersError(`Invalid JSON: ${e.message}`);
    }
  };

  const handleEditProfile = (profile) => {
    setEditingProfileId(profile.id);
    setProfileForm({
      name: profile.name,
      description: profile.description || '',
      url: profile.url,
      use_js: profile.use_js,
      llm_provider: profile.llm_provider || '',
      model: profile.model || '',
      collection_name: profile.collection_name || '',
      collection_description: profile.collection_description || '',
      follow_links: profile.follow_links,
      max_depth: profile.max_depth,
      max_pages: profile.max_pages,
      same_domain_only: profile.same_domain_only,
      headers: profile.headers ? JSON.stringify(profile.headers, null, 2) : '',
    });
    setOpenProfileDialog(true);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <WebIcon color="primary" />
          <Typography variant="h6" color="primary">Web Crawlers</Typography>
        </Box>
        <Button
          variant="contained"
          size="small"
          startIcon={<AddIcon />}
          onClick={() => {
            resetProfileForm();
            setEditingProfileId(null);
            setOpenProfileDialog(true);
          }}
        >
          Add New
        </Button>
      </Box>

      {/* Search */}
      <TextField
        fullWidth
        size="small"
        placeholder="Search crawlers by name..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        sx={{ mb: 2 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon color="action" />
            </InputAdornment>
          ),
        }}
      />

      {/* Loading */}
      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Tiles Grid */}
      <Box sx={{ maxHeight: 500, overflowY: 'auto', pr: 1 }}>
        {filteredProfiles.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              {searchQuery ? 'No crawlers found matching your search.' : 'No crawlers configured. Click "Add New" to create one.'}
            </Typography>
          </Box>
        ) : (
          <Grid container spacing={2}>
            {filteredProfiles.map((profile) => (
              <Grid item xs={12} sm={6} key={profile.id}>
                <Card 
                  sx={{ 
                    height: '100%',
                    border: '1px solid',
                    borderColor: 'divider',
                    '&:hover': {
                      borderColor: 'primary.main',
                      boxShadow: 2,
                    },
                    transition: 'all 0.2s',
                  }}
                >
                  <CardContent sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', flex: 1, mr: 1 }} noWrap>
                        {profile.name}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <Tooltip title="Execute">
                          <IconButton 
                            size="small" 
                            color="success"
                            onClick={(e) => {
                              e.stopPropagation();
                              executeProfileMutation.mutate(profile.id);
                            }}
                            disabled={executeProfileMutation.isLoading}
                          >
                            <PlayIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit">
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
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton 
                            size="small" 
                            color="error"
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteConfirmDialog({ open: true, profileId: profile.id });
                            }}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    
                    {profile.description && (
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }} noWrap>
                        {profile.description}
                      </Typography>
                    )}
                    
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }} noWrap>
                      {profile.url}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {profile.use_js && (
                        <Chip label="JS" size="small" color="info" variant="outlined" />
                      )}
                      {profile.follow_links && (
                        <Chip label="Recursive" size="small" color="secondary" variant="outlined" />
                      )}
                      {profile.collection_name && (
                        <Chip label={profile.collection_name} size="small" variant="outlined" />
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Box>

      {/* Execution status */}
      {executeProfileMutation.isLoading && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
            Crawling in progress...
          </Typography>
        </Box>
      )}

      {executeProfileMutation.isSuccess && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Crawl completed successfully!
        </Alert>
      )}

      {executeProfileMutation.isError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {executeProfileMutation.error?.response?.data?.detail || 'Crawl failed'}
        </Alert>
      )}

      {/* Create/Edit Profile Dialog */}
      <Dialog open={openProfileDialog} onClose={() => setOpenProfileDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>{editingProfileId ? 'Edit Crawler' : 'Create New Crawler'}</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Name"
              value={profileForm.name}
              onChange={(e) => setProfileForm({ ...profileForm, name: e.target.value })}
              sx={{ mb: 2 }}
              required
            />
            <TextField
              fullWidth
              label="Description"
              value={profileForm.description}
              onChange={(e) => setProfileForm({ ...profileForm, description: e.target.value })}
              multiline
              rows={2}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="URL"
              value={profileForm.url}
              onChange={(e) => setProfileForm({ ...profileForm, url: e.target.value })}
              sx={{ mb: 2 }}
              required
              placeholder="https://example.com"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={profileForm.use_js}
                  onChange={(e) => setProfileForm({ ...profileForm, use_js: e.target.checked })}
                />
              }
              label="Use JavaScript Rendering"
              sx={{ mb: 2, display: 'block' }}
            />
            
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Recursive Crawling
            </Typography>
            
            <FormControlLabel
              control={
                <Switch
                  checked={profileForm.follow_links}
                  onChange={(e) => setProfileForm({ ...profileForm, follow_links: e.target.checked })}
                />
              }
              label="Follow Links"
              sx={{ mb: 2, display: 'block' }}
            />
            {profileForm.follow_links && (
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Max Depth"
                    value={profileForm.max_depth}
                    onChange={(e) => setProfileForm({ ...profileForm, max_depth: parseInt(e.target.value) || 3 })}
                    inputProps={{ min: 1, max: 10 }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Max Pages"
                    value={profileForm.max_pages}
                    onChange={(e) => setProfileForm({ ...profileForm, max_pages: parseInt(e.target.value) || 50 })}
                    inputProps={{ min: 1, max: 1000 }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={profileForm.same_domain_only}
                        onChange={(e) => setProfileForm({ ...profileForm, same_domain_only: e.target.checked })}
                      />
                    }
                    label="Same Domain Only"
                  />
                </Grid>
              </Grid>
            )}
            
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              LLM Settings (Optional)
            </Typography>
            
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <FormControl fullWidth>
                  <InputLabel>LLM Provider</InputLabel>
                  <Select
                    value={profileForm.llm_provider}
                    onChange={(e) => setProfileForm({ ...profileForm, llm_provider: e.target.value })}
                    label="LLM Provider"
                  >
                    <MenuItem value="">Default</MenuItem>
                    <MenuItem value="gemini">Gemini</MenuItem>
                    <MenuItem value="qwen">Qwen</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Model"
                  value={profileForm.model}
                  onChange={(e) => setProfileForm({ ...profileForm, model: e.target.value })}
                />
              </Grid>
            </Grid>
            
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Collection Settings
            </Typography>
            
            <TextField
              fullWidth
              label="Collection Name"
              value={profileForm.collection_name}
              onChange={(e) => setProfileForm({ ...profileForm, collection_name: e.target.value })}
              sx={{ mb: 2 }}
              helperText="Leave empty to auto-generate"
            />
            <TextField
              fullWidth
              label="Collection Description"
              value={profileForm.collection_description}
              onChange={(e) => setProfileForm({ ...profileForm, collection_description: e.target.value })}
              multiline
              rows={2}
              sx={{ mb: 2 }}
            />
            
            <TextField
              fullWidth
              label="Headers (JSON)"
              value={profileForm.headers}
              onChange={(e) => setProfileForm({ ...profileForm, headers: e.target.value })}
              multiline
              rows={3}
              sx={{ mb: 2 }}
              placeholder='{"Authorization": "Bearer token"}'
              error={!!headersError}
              helperText={headersError || "Optional JSON object with HTTP headers"}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenProfileDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={editingProfileId ? handleUpdateProfile : handleCreateProfile}
            disabled={(createProfileMutation.isLoading || updateProfileMutation.isLoading) || !profileForm.name.trim() || !profileForm.url.trim()}
          >
            {editingProfileId ? (updateProfileMutation.isLoading ? 'Updating...' : 'Update') : (createProfileMutation.isLoading ? 'Creating...' : 'Create')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteConfirmDialog.open} onClose={() => setDeleteConfirmDialog({ open: false, profileId: null })}>
        <DialogTitle>Delete Crawler</DialogTitle>
        <DialogContent>
          <Typography>Are you sure you want to delete this crawler profile? This action cannot be undone.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmDialog({ open: false, profileId: null })}>Cancel</Button>
          <Button
            variant="contained"
            color="error"
            onClick={() => deleteProfileMutation.mutate(deleteConfirmDialog.profileId)}
            disabled={deleteProfileMutation.isLoading}
          >
            {deleteProfileMutation.isLoading ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Request Tools Panel Component
const RequestToolsPanel = ({ isActive }) => {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState('');
  const { data: requests = [], isLoading } = useQuery('request-tools', api.getRequestTools, { staleTime: 5 * 60 * 1000 });
  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [editingRequestId, setEditingRequestId] = useState(null);
  const [openResponseModal, setOpenResponseModal] = useState(false);
  const [responseData, setResponseData] = useState(null);
  const [validationError, setValidationError] = useState('');
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState({ open: false, requestId: null });
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

  const filteredRequests = useMemo(() => {
    if (!searchQuery.trim()) return requests;
    const query = searchQuery.toLowerCase();
    return requests.filter(r => 
      r.name?.toLowerCase().includes(query) || 
      r.description?.toLowerCase().includes(query) ||
      r.url?.toLowerCase().includes(query)
    );
  }, [requests, searchQuery]);

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
      setDeleteConfirmDialog({ open: false, requestId: null });
    },
  });

  const executeMutation = useMutation(api.executeRequestTool, {
    onSuccess: (data) => {
      queryClient.invalidateQueries('request-tools');
      setResponseData(data);
      setOpenResponseModal(true);
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
    setValidationError('');
  };

  const handleCreate = () => {
    try {
      setValidationError('');
      const payload = {
        ...createForm,
        headers: parseJsonOrEmpty(headersText),
        params: parseJsonOrEmpty(paramsText),
        body: createForm.body ? (tryParseJson(createForm.body) || createForm.body) : null,
      };
      createMutation.mutate(payload);
    } catch (e) {
      setValidationError(e.message);
    }
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
    try {
      setValidationError('');
      const payload = {
        ...createForm,
        headers: parseJsonOrEmpty(headersText),
        params: parseJsonOrEmpty(paramsText),
        body: createForm.body ? (tryParseJson(createForm.body) || createForm.body) : null,
      };
      updateMutation.mutate({ requestId: editingRequestId, payload });
    } catch (e) {
      setValidationError(e.message);
    }
  };

  const parseJsonOrEmpty = (text) => {
    if (!text || !text.trim()) return {};
    try {
      const parsed = JSON.parse(text);
      if (typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('Must be a JSON object');
      }
      return parsed;
    } catch (e) {
      throw new Error(`Invalid JSON: ${e.message}`);
    }
  };

  const tryParseJson = (text) => {
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  };

  const getMethodColor = (method) => {
    const colors = {
      GET: 'success',
      POST: 'primary',
      PUT: 'warning',
      DELETE: 'error',
      PATCH: 'info',
    };
    return colors[method] || 'default';
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ApiIcon color="primary" />
          <Typography variant="h6" color="primary">REST API Requests</Typography>
        </Box>
        <Button
          variant="contained"
          size="small"
          startIcon={<AddIcon />}
          onClick={() => {
            resetForm();
            setOpenCreateDialog(true);
          }}
        >
          Add New
        </Button>
      </Box>

      {/* Search */}
      <TextField
        fullWidth
        size="small"
        placeholder="Search requests by name..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        sx={{ mb: 2 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon color="action" />
            </InputAdornment>
          ),
        }}
      />

      {/* Loading */}
      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Tiles Grid */}
      <Box sx={{ maxHeight: 500, overflowY: 'auto', pr: 1 }}>
        {filteredRequests.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              {searchQuery ? 'No requests found matching your search.' : 'No requests configured. Click "Add New" to create one.'}
            </Typography>
          </Box>
        ) : (
          <Grid container spacing={2}>
            {filteredRequests.map((req) => (
              <Grid item xs={12} sm={6} key={req.id}>
                <Card 
                  sx={{ 
                    height: '100%',
                    border: '1px solid',
                    borderColor: 'divider',
                    '&:hover': {
                      borderColor: 'primary.main',
                      boxShadow: 2,
                    },
                    transition: 'all 0.2s',
                  }}
                >
                  <CardContent sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', flex: 1, mr: 1 }} noWrap>
                        {req.name}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <Tooltip title="Execute">
                          <IconButton 
                            size="small" 
                            color="success"
                            onClick={(e) => {
                              e.stopPropagation();
                              executeMutation.mutate(req.id);
                            }}
                            disabled={executeMutation.isLoading}
                          >
                            <SendIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit">
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
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton 
                            size="small" 
                            color="error"
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteConfirmDialog({ open: true, requestId: req.id });
                            }}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    
                    <Box sx={{ display: 'flex', gap: 0.5, mb: 1, alignItems: 'center' }}>
                      <Chip 
                        label={req.request_type?.toUpperCase() || 'HTTP'} 
                        size="small" 
                        color={req.request_type === 'http' ? 'primary' : 'secondary'}
                        variant="outlined"
                      />
                      {req.method && (
                        <Chip 
                          label={req.method} 
                          size="small" 
                          color={getMethodColor(req.method)}
                        />
                      )}
                    </Box>
                    
                    {req.last_executed_at && (
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                        Last: {new Date(req.last_executed_at).toLocaleString()}
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Box>

      {/* Execution status */}
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
          {executeMutation.error?.response?.data?.detail || 'Execution failed'}
        </Alert>
      )}

      {/* Create Dialog */}
      <Dialog open={openCreateDialog} onClose={() => setOpenCreateDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New Request</DialogTitle>
        <DialogContent>
          {validationError && (
            <Alert severity="error" sx={{ mb: 2, mt: 1 }} onClose={() => setValidationError('')}>
              {validationError}
            </Alert>
          )}
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
      <Dialog open={openEditDialog} onClose={() => {
        setOpenEditDialog(false);
        setValidationError('');
      }} maxWidth="md" fullWidth>
        <DialogTitle>Edit Request</DialogTitle>
        <DialogContent>
          {validationError && (
            <Alert severity="error" sx={{ mb: 2, mt: 1 }} onClose={() => setValidationError('')}>
              {validationError}
            </Alert>
          )}
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

      {/* Response Modal */}
      <Dialog 
        open={openResponseModal} 
        onClose={() => setOpenResponseModal(false)} 
        maxWidth="lg" 
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">Response Details</Typography>
            {responseData && (
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <Chip
                  label={responseData.success ? 'Success' : 'Failed'}
                  color={responseData.success ? 'success' : 'error'}
                  size="small"
                />
                {responseData.status_code && (
                  <Chip
                    label={`Status: ${responseData.status_code}`}
                    size="small"
                    variant="outlined"
                  />
                )}
              </Box>
            )}
          </Box>
        </DialogTitle>
        <DialogContent>
          {responseData && (
            <Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2">
                  <strong>Request:</strong> {responseData.request_name}
                </Typography>
                {responseData.execution_time !== undefined && (
                  <Typography variant="body2">
                    <strong>Execution Time:</strong> {responseData.execution_time.toFixed(3)}s
                  </Typography>
                )}
              </Box>

              {responseData.error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {responseData.error}
                </Alert>
              )}

              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Response Body
              </Typography>
              <Paper sx={{ 
                p: 2, 
                bgcolor: 'background.paper', 
                border: '1px solid',
                borderColor: 'divider',
                maxHeight: 400, 
                overflow: 'auto',
              }}>
                <pre style={{ margin: 0, fontSize: '0.875rem', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {typeof responseData.response_data === 'object' 
                    ? JSON.stringify(responseData.response_data, null, 2)
                    : responseData.response_data || 'No response body'}
                </pre>
              </Paper>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenResponseModal(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmDialog.open}
        onClose={() => setDeleteConfirmDialog({ open: false, requestId: null })}
      >
        <DialogTitle>Delete Request</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this request? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmDialog({ open: false, requestId: null })}>
            Cancel
          </Button>
          <Button 
            onClick={() => deleteMutation.mutate(deleteConfirmDialog.requestId)} 
            color="error" 
            variant="contained"
            disabled={deleteMutation.isLoading}
          >
            {deleteMutation.isLoading ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Request Form Component
const RequestForm = ({ form, setForm, headersText, setHeadersText, paramsText, setParamsText }) => {
  return (
    <Box sx={{ mt: 1 }}>
      <TextField
        fullWidth
        label="Request Name"
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
        placeholder='{"Content-Type": "application/json"}'
        helperText="JSON object with key-value pairs"
      />
      <TextField
        fullWidth
        label="Query Parameters (JSON)"
        value={paramsText}
        onChange={(e) => setParamsText(e.target.value)}
        multiline
        rows={2}
        sx={{ mb: 2 }}
        placeholder='{"param1": "value1"}'
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
      />
    </Box>
  );
};

export default Crawler;
