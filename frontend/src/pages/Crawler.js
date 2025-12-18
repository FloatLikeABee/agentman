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
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { useMutation } from 'react-query';
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
        {/* Input Section */}
        <Grid item xs={12} md={8}>
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

        {/* Results Section */}
        <Grid item xs={12} md={4}>
          {crawlMutation.isSuccess && crawlMutation.data && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CheckCircleIcon color="success" />
                  Crawl Results
                </Typography>

                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Collection Name
                  </Typography>
                  <Chip
                    label={crawlMutation.data.collection_name || 'N/A'}
                    color="primary"
                    sx={{ mb: 1 }}
                  />

                  <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 2 }}>
                    Collection Description
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    {crawlMutation.data.collection_description || 'N/A'}
                  </Typography>

                  {crawlMutation.data.extracted_data && (
                    <>
                      <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 2 }}>
                        Extracted Information
                      </Typography>
                      <Box sx={{ mt: 1 }}>
                        <Chip
                          label={`${crawlMutation.data.extracted_data.main_topics_count || 0} Topics`}
                          size="small"
                          sx={{ mr: 1, mb: 1 }}
                        />
                        <Chip
                          label={`${crawlMutation.data.extracted_data.key_points_count || 0} Key Points`}
                          size="small"
                          sx={{ mr: 1, mb: 1 }}
                        />
                      </Box>
                      {crawlMutation.data.extracted_data.title && (
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          <strong>Title:</strong> {crawlMutation.data.extracted_data.title}
                        </Typography>
                      )}
                    </>
                  )}

                  {crawlMutation.data.raw_file && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                      Raw file: {crawlMutation.data.raw_file.split('/').pop()}
                    </Typography>
                  )}

                  {crawlMutation.data.extracted_file && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                      Extracted file: {crawlMutation.data.extracted_file.split('/').pop()}
                    </Typography>
                  )}
                </Box>
              </CardContent>
            </Card>
          )}

          {crawlMutation.isError && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ErrorIcon color="error" />
                  Error
                </Typography>
                <Typography variant="body2" color="error" sx={{ mt: 1 }}>
                  {crawlMutation.error?.response?.data?.detail || crawlMutation.error?.message || 'Unknown error occurred'}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default Crawler;

