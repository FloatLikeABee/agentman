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
  Language as CrawlerIcon,
  CompareArrows as EqualizerIcon,
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
      crawler: 'error',
      equalizer: 'info',
      custom: 'default',
    };
    return colors[type] || 'default';
  };

  const getToolUsageExample = (toolId) => {
    const examples = {
      web_search: 'latest developments in artificial intelligence 2024',
      wikipedia: 'artificial intelligence',
      calculator: '(25 * 4) + (100 / 2)',
      email: 'to:user@example.com,subject:Meeting Reminder,body:Don\'t forget about our meeting tomorrow at 2 PM.',
      financial: 'stock price AAPL',
      crawler: 'url:https://example.com,collection_name:my_collection,description:Data from example.com',
      equalizer: 'scenario:I need to decide between working remotely or in the office. Consider factors like productivity, work-life balance, and team collaboration.',
    };
    return examples[toolId] || null;
  };

  const getToolIcon = (toolType) => {
    if (toolType === 'crawler') return <CrawlerIcon />;
    if (toolType === 'equalizer') return <EqualizerIcon />;
    return <ToolIcon />;
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
        {tools?.map((tool) => {
          const usageExample = getToolUsageExample(tool.id);
          return (
            <Grid item xs={12} md={6} lg={4} key={tool.id}>
              <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getToolIcon(tool.tool_type)}
                      <Typography variant="h6">{tool.name}</Typography>
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
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                    {tool.description}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mb: usageExample ? 1.5 : 0, flexWrap: 'wrap' }}>
                    <Chip 
                      label={tool.tool_type} 
                      size="small" 
                      color={getToolTypeColor(tool.tool_type)}
                    />
                    <Chip 
                      label={tool.is_active ? 'Active' : 'Inactive'} 
                      size="small" 
                      color={tool.is_active ? 'success' : 'default'}
                    />
                  </Box>
                  {usageExample && (
                    <Box sx={{ mt: 1.5, p: 1.5, bgcolor: 'grey.50', borderRadius: 1, border: '1px solid', borderColor: 'grey.300' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 'bold', display: 'block', mb: 0.5 }}>
                        Usage Example:
                      </Typography>
                      <Typography 
                        variant="caption" 
                        sx={{ 
                          fontFamily: 'monospace', 
                          fontSize: '0.75rem',
                          wordBreak: 'break-all',
                          display: 'block',
                          color: 'text.primary'
                        }}
                      >
                        {usageExample}
                      </Typography>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          );
        })}
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