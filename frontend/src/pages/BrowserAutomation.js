import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Paper,
  Divider,
  IconButton,
  Tooltip,
  Chip,
  FormControlLabel,
  Switch,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  UploadFile as UploadFileIcon,
  Clear as ClearIcon,
  ContentCopy as CopyIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useQuery, useMutation } from 'react-query';
import api from '../services/api';

const BrowserAutomation = () => {
  const [prompt, setPrompt] = useState('');
  const [selectedProvider, setSelectedProvider] = useState('qwen');
  const [selectedModel, setSelectedModel] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [headless, setHeadless] = useState(false); // Default to visible browser
  const [browserBridgeUrl, setBrowserBridgeUrl] = useState('ws://localhost:8765');

  // Fetch providers
  const { data: providersData = { providers: [] } } = useQuery('providers', api.getProviders);
  const providers = providersData.providers || [];

  // Fetch models and transform to {provider: [model1, model2, ...]} format
  const { data: modelsData = [] } = useQuery('models', api.getModels);
  const models = useMemo(() => {
    const modelsByProvider = {};
    (Array.isArray(modelsData) ? modelsData : []).forEach((model) => {
      const provider = model.provider;
      if (!modelsByProvider[provider]) {
        modelsByProvider[provider] = [];
      }
      if (model.name && !modelsByProvider[provider].includes(model.name)) {
        modelsByProvider[provider].push(model.name);
      }
    });
    return modelsByProvider;
  }, [modelsData]);

  // Set default model when provider changes
  useEffect(() => {
    if (models[selectedProvider] && models[selectedProvider].length > 0) {
      setSelectedModel(models[selectedProvider][0]);
    }
  }, [selectedProvider, models]);

  // Execute browser automation mutation
  const executeMutation = useMutation(
    (data) => api.executeBrowserAutomation(data),
    {
      onSuccess: (data) => {
        setResult(data.result || data);
        setIsExecuting(false);
        setError(null);
      },
      onError: (err) => {
        setError(err.response?.data?.detail || 'Failed to execute browser automation');
        setIsExecuting(false);
      },
    }
  );

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPrompt(e.target.result);
      };
      reader.readAsText(file);
    }
  };

  const handleExecute = () => {
    if (!prompt.trim()) {
      setError('Please enter instructions or upload a prompt file');
      return;
    }
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }
    
    setError(null);
    setResult(null);
    setIsExecuting(true);
    
    executeMutation.mutate({
      instructions: prompt,
      provider: selectedProvider,
      model: selectedModel,
      max_steps: 20,
      headless: headless,
      browser_bridge_url: browserBridgeUrl.trim() || 'ws://localhost:8765',
    });
  };

  const handleClear = () => {
    setPrompt('');
    setResult(null);
    setError(null);
    setUploadedFileName(null);
  };

  const handleCopyResult = () => {
    if (result) {
      navigator.clipboard.writeText(result);
    }
  };

  const handleDownloadResult = () => {
    if (result) {
      const blob = new Blob([result], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `browser-automation-result-${new Date().toISOString()}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: 1,
        color: '#9d4edd',
        fontWeight: 600,
      }}>
        🌐 Browser Automation
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Control a web browser using AI. Provide natural language instructions and the AI agent will execute them step by step.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Main Input Section */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={3}>
            {/* Prompt Input Section */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'medium' }}>
                  Instructions
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <input
                    accept=".txt,.md"
                    style={{ display: 'none' }}
                    id="upload-prompt-file"
                    type="file"
                    onChange={handleFileUpload}
                  />
                  <label htmlFor="upload-prompt-file">
                    <Tooltip title="Upload prompt file">
                      <IconButton component="span" size="small" color="primary">
                        <UploadFileIcon />
                      </IconButton>
                    </Tooltip>
                  </label>
                  {uploadedFileName && (
                    <Chip 
                      label={uploadedFileName} 
                      size="small" 
                      onDelete={() => {
                        setUploadedFileName(null);
                        setPrompt('');
                      }}
                    />
                  )}
                  <Tooltip title="Clear">
                    <IconButton size="small" onClick={handleClear}>
                      <ClearIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              <TextField
                fullWidth
                multiline
                rows={15}
                label="Browser Automation Instructions"
                placeholder="Enter detailed instructions for the browser automation...&#10;&#10;Examples:&#10;- Go to google.com and search for 'Python programming'&#10;- Navigate to example.com, fill the contact form with name 'John Doe' and email 'john@example.com', then submit&#10;- Open github.com, search for 'langchain', and get the first 5 repository names&#10;- Go to news.ycombinator.com, scroll down, and take a screenshot"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&:hover fieldset': {
                      borderColor: '#9d4edd',
                    },
                    fontFamily: 'monospace',
                    fontSize: '0.9rem',
                  },
                }}
              />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                {prompt.length} characters
              </Typography>
            </Grid>

            {/* Provider and Model Selection */}
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>AI Provider</InputLabel>
                <Select
                  value={selectedProvider}
                  label="AI Provider"
                  onChange={(e) => setSelectedProvider(e.target.value)}
                >
                  {providers.map((provider) => (
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
                  value={selectedModel}
                  label="Model"
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {(models[selectedProvider] || []).map((model) => (
                    <MenuItem key={model} value={model}>
                      {model}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Browser Mode Toggle */}
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={!headless}
                    onChange={(e) => setHeadless(!e.target.checked)}
                    color="primary"
                  />
                }
                label={
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                      {headless ? '🔇 Headless Mode (Background)' : '👁️ Visible Browser Mode'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {headless 
                        ? 'Browser runs in background (faster, no visual feedback)' 
                        : 'Browser window will open so you can see actions in real-time'}
                    </Typography>
                  </Box>
                }
              />
            </Grid>

            {/* Local browser bridge (optional) */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                size="small"
                label="Browser Bridge URL (optional)"
                placeholder="ws://localhost:8765 — use your local browser when AI is in the cloud"
                value={browserBridgeUrl}
                onChange={(e) => setBrowserBridgeUrl(e.target.value)}
                helperText="Run python browser_bridge.py on your machine, then enter its WebSocket URL so the cloud AI controls your local Chrome."
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&:hover fieldset': { borderColor: '#9d4edd' },
                  },
                }}
              />
            </Grid>

            {/* Execute Button */}
            <Grid item xs={12}>
              <Button
                fullWidth
                variant="contained"
                size="large"
                startIcon={isExecuting ? <CircularProgress size={24} color="inherit" /> : <PlayIcon />}
                onClick={handleExecute}
                disabled={isExecuting || !prompt.trim() || !selectedModel}
                sx={{
                  py: 1.5,
                  fontSize: '1.1rem',
                  background: 'linear-gradient(135deg, #9d4edd 0%, #7b2cbf 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #c77dff 0%, #9d4edd 100%)',
                  },
                }}
              >
                {isExecuting ? 'Executing...' : 'Execute Browser Automation'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Results Section */}
      {result && (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                📋 Execution Results
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Tooltip title="Copy result">
                  <IconButton size="small" onClick={handleCopyResult}>
                    <CopyIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Download result">
                  <IconButton size="small" onClick={handleDownloadResult}>
                    <DownloadIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <Paper 
              variant="outlined" 
              sx={{ 
                p: 2, 
                bgcolor: 'rgba(157, 78, 221, 0.05)',
                borderColor: 'rgba(157, 78, 221, 0.2)',
                maxHeight: 600,
                overflowY: 'auto',
              }}
            >
              <Typography 
                variant="body2" 
                sx={{ 
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'monospace',
                  fontSize: '0.9rem',
                  lineHeight: 1.6,
                }}
              >
                {result}
              </Typography>
            </Paper>
          </CardContent>
        </Card>
      )}

      {/* Info Card */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            💡 Capabilities
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Typography variant="body2" color="text.secondary">
                • Navigate to websites
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Typography variant="body2" color="text.secondary">
                • Click buttons and links
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Typography variant="body2" color="text.secondary">
                • Fill forms and input fields
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Typography variant="body2" color="text.secondary">
                • Extract text and data
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Typography variant="body2" color="text.secondary">
                • Take screenshots
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Typography variant="body2" color="text.secondary">
                • Scroll pages
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Typography variant="body2" color="text.secondary">
                • Wait for elements
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Typography variant="body2" color="text.secondary">
                • Select dropdown options
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default BrowserAutomation;
