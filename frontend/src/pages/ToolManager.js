import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Switch,
  FormControlLabel,
  Grid,
  Chip,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  Build as ToolIcon,
  Settings as SettingsIcon,
  Code as CodeIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';

const ToolManager = () => {
  const [openConfigDialog, setOpenConfigDialog] = useState(false);
  const [selectedTool, setSelectedTool] = useState(null);
  const [configJson, setConfigJson] = useState('');
  const [jsonError, setJsonError] = useState('');

  const queryClient = useQueryClient();
  const { data: tools, isLoading } = useQuery('tools', api.getTools);

  const updateToolMutation = useMutation(api.updateToolConfig, {
    onSuccess: () => {
      queryClient.invalidateQueries('tools');
      setOpenConfigDialog(false);
      setSelectedTool(null);
      setConfigJson('');
      setJsonError('');
    },
  });

  const handleUpdateTool = () => {
    if (selectedTool && !jsonError) {
      try {
        const config = JSON.parse(configJson);
        updateToolMutation.mutate({
          tool_id: selectedTool.id,
          config: {
            ...selectedTool,
            config: config,
          },
        });
      } catch (error) {
        setJsonError('Invalid JSON format');
      }
    }
  };

  const getToolTypeColor = (type) => {
    const colors = {
      email: 'primary',
      web_search: 'secondary',
      calculator: 'success',
      financial: 'warning',
      wikipedia: 'info',
      custom: 'default',
    };
    return colors[type] || 'default';
  };

  if (isLoading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Tool Manager</Typography>
      </Box>

      {/* Tools List */}
      <Grid container spacing={3}>
        {tools?.map((tool) => (
          <Grid item xs={12} md={6} lg={4} key={tool.id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Box>
                    <Typography variant="h6">{tool.name}</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {tool.description}
                    </Typography>
                    <Chip 
                      label={tool.tool_type} 
                      size="small" 
                      color={getToolTypeColor(tool.tool_type)}
                      sx={{ mr: 1 }}
                    />
                    <Chip 
                      label={tool.is_active ? 'Active' : 'Inactive'} 
                      size="small" 
                      color={tool.is_active ? 'success' : 'default'}
                    />
                  </Box>
                  <Button
                    size="small"
                    startIcon={<SettingsIcon />}
                    onClick={() => {
                      setSelectedTool(tool);
                      setConfigJson(JSON.stringify(tool.config || {}, null, 2));
                      setJsonError('');
                      setOpenConfigDialog(true);
                    }}
                  >
                    Config
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Tool Configuration Dialog */}
      <Dialog open={openConfigDialog} onClose={() => setOpenConfigDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ pb: 1 }}>Configure Tool: {selectedTool?.name}</DialogTitle>
        <DialogContent sx={{ pt: 1 }}>
          {updateToolMutation.isError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to update tool configuration. Please try again.
            </Alert>
          )}
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Tool Information
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Type: {selectedTool?.tool_type}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Description: {selectedTool?.description}
              </Typography>
            </Grid>
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={selectedTool?.is_active || false}
                    onChange={(e) => setSelectedTool({ 
                      ...selectedTool, 
                      is_active: e.target.checked 
                    })}
                  />
                }
                label="Active"
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CodeIcon color="primary" />
                Configuration (JSON)
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={12}
                value={configJson}
                onChange={(e) => {
                  setConfigJson(e.target.value);
                  try {
                    JSON.parse(e.target.value);
                    setJsonError('');
                  } catch (error) {
                    setJsonError('Invalid JSON format');
                  }
                }}
                error={!!jsonError}
                helperText={jsonError}
                placeholder='{"key": "value"}'
                sx={{
                  fontFamily: 'monospace',
                  '& .MuiInputBase-input': {
                    fontFamily: 'monospace',
                  },
                }}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenConfigDialog(false)}>Cancel</Button>
          <Button
            onClick={handleUpdateTool}
            variant="contained"
            disabled={updateToolMutation.isLoading || !!jsonError}
          >
            {updateToolMutation.isLoading ? 'Updating...' : 'Update Tool'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ToolManager; 