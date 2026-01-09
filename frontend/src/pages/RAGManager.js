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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  IconButton,
  List,
  ListItem,
  Alert,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Search as SearchIcon,
  UploadFile as UploadFileIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';
import ReactJson from 'react-json-view';

const RAGManager = () => {
  const [openAddDialog, setOpenAddDialog] = useState(false);
  const [openQueryDialog, setOpenQueryDialog] = useState(false);
  const [selectedCollection, setSelectedCollection] = useState('');
  const [queryText, setQueryText] = useState('');
  const [queryResults, setQueryResults] = useState([]);
  const [queryError, setQueryError] = useState('');
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState({ open: false, collectionName: null });
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    format: 'json',
    content: '',
    tags: [],
    metadata: {},
  });
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileError, setFileError] = useState('');
  const [importMode, setImportMode] = useState('manual'); // 'manual' or 'file'

  const queryClient = useQueryClient();
  const { data: collections, isLoading } = useQuery('collections', api.getRAGCollections, { staleTime: 5 * 60 * 1000 }); // Cache for 5 minutes

  const addDataMutation = useMutation(api.addRAGData, {
    onSuccess: () => {
      queryClient.invalidateQueries('collections');
      setOpenAddDialog(false);
      setFormData({
        name: '',
        description: '',
        format: 'json',
        content: '',
        tags: [],
        metadata: {},
      });
      setSelectedFile(null);
      setFileError('');
      setImportMode('manual');
    },
  });

  const deleteCollectionMutation = useMutation(api.deleteRAGCollection, {
    onSuccess: () => {
      queryClient.invalidateQueries('collections');
      setDeleteConfirmDialog({ open: false, collectionName: null });
    },
  });

  const handleAddData = () => {
    addDataMutation.mutate({
      collection_name: formData.name,
      data_input: formData,
    });
  };

  const handleQueryCollection = async () => {
    try {
      setQueryError('');
      const results = await api.queryRAGCollection(selectedCollection, queryText);
      setQueryResults(results.results || []);
    } catch (error) {
      console.error('Query error:', error);
      setQueryError('Failed to query collection. Please try again.');
    }
  };

  const handleDeleteCollection = (collectionName) => {
    setDeleteConfirmDialog({ open: true, collectionName });
  };

  const handleDeleteConfirm = () => {
    deleteCollectionMutation.mutate(deleteConfirmDialog.collectionName);
  };

  const handleFileSelect = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setFileError('');

    // Get file extension
    const fileName = file.name.toLowerCase();
    let detectedFormat = 'txt';
    
    if (fileName.endsWith('.json')) {
      detectedFormat = 'json';
    } else if (fileName.endsWith('.csv')) {
      detectedFormat = 'csv';
    } else if (fileName.endsWith('.txt')) {
      detectedFormat = 'txt';
    }

    // Auto-set collection name from filename if not set
    if (!formData.name.trim()) {
      const nameWithoutExt = file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_');
      setFormData(prev => ({ ...prev, name: nameWithoutExt }));
    }

    // Read file content
    try {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        
        // Validate and parse based on format
        if (detectedFormat === 'json') {
          try {
            // Validate JSON
            JSON.parse(content);
            setFormData(prev => ({
              ...prev,
              format: 'json',
              content: content,
            }));
          } catch (error) {
            setFileError('Invalid JSON file. Please check the file format.');
            setSelectedFile(null);
          }
        } else if (detectedFormat === 'csv') {
          // CSV can be sent as-is, backend will parse it
          setFormData(prev => ({
            ...prev,
            format: 'csv',
            content: content,
          }));
        } else {
          // TXT file
          setFormData(prev => ({
            ...prev,
            format: 'txt',
            content: content,
          }));
        }
      };
      
      reader.onerror = () => {
        setFileError('Error reading file. Please try again.');
        setSelectedFile(null);
      };
      
      reader.readAsText(file);
    } catch (error) {
      setFileError('Error processing file. Please try again.');
      setSelectedFile(null);
    }
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    setFileError('');
    setFormData(prev => ({ ...prev, content: '' }));
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">RAG Manager</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpenAddDialog(true)}
        >
          Add Data
        </Button>
      </Box>

      {/* Collections List */}
      <Grid container spacing={3}>
        {collections?.map((collection) => (
          <Grid item xs={12} md={6} lg={4} key={collection.name}>
            <Card sx={{ boxShadow: 2, transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' }, height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ p: 3, display: 'flex', flexDirection: 'column', flexGrow: 1, minHeight: 0 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2, gap: 1, minWidth: 0 }}>
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography 
                      variant="h6" 
                      sx={{ 
                        mb: 1,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        pr: 1
                      }}
                      title={collection.name}
                    >
                      {collection.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {collection.count} documents
                    </Typography>
                    {collection.metadata?.description && (
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          mb: 2,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 3,
                          WebkitBoxOrient: 'vertical',
                          minHeight: '4.5em',
                          maxHeight: '4.5em',
                          lineHeight: 1.5
                        }}
                        title={collection.metadata.description}
                      >
                        {collection.metadata.description}
                      </Typography>
                    )}
                  </Box>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={() => handleDeleteCollection(collection.name)}
                    sx={{ 
                      bgcolor: 'error.light', 
                      '&:hover': { bgcolor: 'error.main', color: 'white' },
                      flexShrink: 0,
                      ml: 'auto'
                    }}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Box>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<SearchIcon />}
                  onClick={() => {
                    setSelectedCollection(collection.name);
                    setOpenQueryDialog(true);
                  }}
                  fullWidth
                  sx={{ mt: 'auto' }}
                >
                  Query Collection
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Add Data Dialog */}
      <Dialog 
        open={openAddDialog} 
        onClose={() => {
          setOpenAddDialog(false);
          setImportMode('manual');
          setSelectedFile(null);
          setFileError('');
        }} 
        maxWidth="md" 
        fullWidth
      >
        <DialogTitle sx={{ pb: 1 }}>Add RAG Data</DialogTitle>
        <DialogContent sx={{ pt: 1 }}>
          {addDataMutation.isError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to add data. Please check your inputs and try again.
            </Alert>
          )}
          <Grid container spacing={3}>
            {/* Basic Information */}
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom color="primary">
                Collection Details
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Collection Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value.replace(/[^a-zA-Z0-9_-]/g, '_') })}
                required
                error={!formData.name.trim()}
                helperText={!formData.name.trim() ? 'Collection name is required (spaces will be replaced with underscores)' : 'Spaces will be replaced with underscores'}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Data Format</InputLabel>
                <Select
                  value={formData.format}
                  onChange={(e) => setFormData({ ...formData, format: e.target.value })}
                >
                  <MenuItem value="json">JSON</MenuItem>
                  <MenuItem value="csv">CSV</MenuItem>
                  <MenuItem value="txt">Text</MenuItem>
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

            {/* Data Content */}
            <Grid item xs={12} sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" color="primary">
                  Data Content
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    size="small"
                    variant={importMode === 'manual' ? 'contained' : 'outlined'}
                    onClick={() => {
                      setImportMode('manual');
                      setSelectedFile(null);
                      setFileError('');
                    }}
                  >
                    Manual Input
                  </Button>
                  <Button
                    size="small"
                    variant={importMode === 'file' ? 'contained' : 'outlined'}
                    startIcon={<UploadFileIcon />}
                    onClick={() => setImportMode('file')}
                    component="label"
                  >
                    Import File
                    <input
                      type="file"
                      hidden
                      accept=".txt,.json,.csv"
                      onChange={handleFileSelect}
                    />
                  </Button>
                </Box>
              </Box>
            </Grid>
            
            {importMode === 'file' && (
              <Grid item xs={12}>
                <Box sx={{ mb: 2 }}>
                  {selectedFile ? (
                    <Box sx={{ 
                      p: 2, 
                      border: '1px solid', 
                      borderColor: 'divider', 
                      borderRadius: 1,
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <Box>
                        <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                          {selectedFile.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {(selectedFile.size / 1024).toFixed(2)} KB
                        </Typography>
                      </Box>
                      <Button size="small" onClick={handleClearFile}>
                        Clear
                      </Button>
                    </Box>
                  ) : (
                    <Box
                      sx={{
                        p: 3,
                        border: '2px dashed',
                        borderColor: 'divider',
                        borderRadius: 1,
                        textAlign: 'center',
                        cursor: 'pointer',
                        '&:hover': {
                          borderColor: 'primary.main',
                          bgcolor: 'action.hover',
                        },
                      }}
                      component="label"
                    >
                      <input
                        type="file"
                        hidden
                        accept=".txt,.json,.csv"
                        onChange={handleFileSelect}
                      />
                      <UploadFileIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
                      <Typography variant="body1" sx={{ mb: 1 }}>
                        Click to upload or drag and drop
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Supports: TXT, JSON, CSV files
                      </Typography>
                    </Box>
                  )}
                  {fileError && (
                    <Alert severity="error" sx={{ mt: 1 }}>
                      {fileError}
                    </Alert>
                  )}
                </Box>
              </Grid>
            )}
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={10}
                label="Content"
                value={formData.content}
                onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                placeholder={importMode === 'file' ? 'Select a file to import, or enter content manually...' : 'Enter your data here...'}
                required
                error={!formData.content.trim()}
                helperText={!formData.content.trim() ? 'Content is required' : importMode === 'file' && selectedFile ? 'File loaded. You can edit the content if needed.' : ''}
                disabled={importMode === 'file' && !selectedFile && !formData.content.trim()}
              />
            </Grid>

            {/* Metadata */}
            <Grid item xs={12} sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom color="primary">
                Metadata (Optional)
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Tags (comma-separated)"
                value={formData.tags.join(', ')}
                onChange={(e) => setFormData({
                  ...formData,
                  tags: e.target.value.split(',').map(tag => tag.trim()).filter(tag => tag)
                })}
                placeholder="e.g. science, research, data"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions sx={{ p: 3, pt: 1 }}>
          <Button onClick={() => {
            setOpenAddDialog(false);
            setImportMode('manual');
            setSelectedFile(null);
            setFileError('');
          }}>Cancel</Button>
          <Button
            onClick={handleAddData}
            variant="contained"
            disabled={addDataMutation.isLoading || !formData.name.trim() || !formData.content.trim()}
          >
            {addDataMutation.isLoading ? 'Adding Data...' : 'Add Data'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Query Dialog */}
      <Dialog open={openQueryDialog} onClose={() => setOpenQueryDialog(false)} maxWidth="lg" fullWidth>
        <DialogTitle sx={{ pb: 1 }}>Query Collection: {selectedCollection}</DialogTitle>
        <DialogContent sx={{ pt: 1 }}>
          {queryError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {queryError}
            </Alert>
          )}
          <Box sx={{ mb: 3 }}>
            <TextField
              fullWidth
              label="Search Query"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              multiline
              rows={3}
              placeholder="Enter your search query..."
              sx={{ mb: 2 }}
            />
            <Button
              variant="contained"
              onClick={handleQueryCollection}
              size="large"
              fullWidth
              disabled={!queryText.trim()}
            >
              Search Collection
            </Button>
          </Box>

          {queryResults.length > 0 && (
            <Box>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SearchIcon color="primary" />
                Results ({queryResults.length})
              </Typography>
              <List sx={{ maxHeight: 400, overflowY: 'auto' }}>
                {queryResults.map((result, index) => (
                  <ListItem key={index} divider sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                    <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'medium' }}>
                      Result {index + 1}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                      {result.content.length > 300 ? `${result.content.substring(0, 300)}...` : result.content}
                    </Typography>
                    {result.metadata && (
                      <Box sx={{ width: '100%', mt: 1 }}>
                        <ReactJson
                          src={result.metadata}
                          name="Metadata"
                          collapsed={true}
                          displayDataTypes={false}
                          theme={{
                            base00: 'transparent',
                            base01: '#1a0d2e',
                            base02: '#0f0519',
                            base03: '#b0b0b0',
                            base04: '#9d4edd',
                            base05: '#e0e0e0',
                            base06: '#c77dff',
                            base07: '#e0e0e0',
                            base08: '#ff6b35',
                            base09: '#ff6b35',
                            base0A: '#c77dff',
                            base0B: '#00ff88',
                            base0C: '#9d4edd',
                            base0D: '#9d4edd',
                            base0E: '#c77dff',
                            base0F: '#ff6b35',
                          }}
                          style={{ 
                            backgroundColor: 'transparent',
                            fontSize: '0.875rem'
                          }}
                          iconStyle="circle"
                          enableClipboard={false}
                        />
                      </Box>
                    )}
                  </ListItem>
                ))}
              </List>
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ p: 3, pt: 1 }}>
          <Button onClick={() => setOpenQueryDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmDialog.open}
        onClose={() => setDeleteConfirmDialog({ open: false, collectionName: null })}
      >
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete collection "{deleteConfirmDialog.collectionName}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmDialog({ open: false, collectionName: null })}>
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

export default RAGManager; 