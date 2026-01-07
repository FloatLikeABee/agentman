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

  const queryClient = useQueryClient();
  const { data: collections, isLoading } = useQuery('collections', api.getRAGCollections);

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
            <Card sx={{ boxShadow: 2, transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' }, height: '100%' }}>
              <CardContent sx={{ p: 3, minHeight: 180, display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2, flexGrow: 1 }}>
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="h6" sx={{ mb: 1 }}>{collection.name}</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {collection.count} documents
                    </Typography>
                    {collection.metadata?.description && (
                      <Typography variant="body2" sx={{ mb: 2 }}>
                        {collection.metadata.description}
                      </Typography>
                    )}
                  </Box>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={() => handleDeleteCollection(collection.name)}
                    sx={{ bgcolor: 'error.light', '&:hover': { bgcolor: 'error.main', color: 'white' } }}
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
                >
                  Query Collection
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Add Data Dialog */}
      <Dialog open={openAddDialog} onClose={() => setOpenAddDialog(false)} maxWidth="md" fullWidth>
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
              <Typography variant="h6" gutterBottom color="primary">
                Data Content
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={10}
                label="Content"
                value={formData.content}
                onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                placeholder="Enter your data here..."
                required
                error={!formData.content.trim()}
                helperText={!formData.content.trim() ? 'Content is required' : ''}
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
          <Button onClick={() => setOpenAddDialog(false)}>Cancel</Button>
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
                          style={{ backgroundColor: 'transparent' }}
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