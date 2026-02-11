import React, { useState, useMemo, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  ButtonGroup,
  TextField,
  Alert,
  CircularProgress,
  Grid,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
} from '@mui/material';
import { AutoAwesome as GenerateIcon, Code as MarkdownIcon, Html as HtmlIcon, OpenInFull as ExpandIcon, Close as CloseIcon } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { useQuery, useMutation } from 'react-query';
import api from '../services/api';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function ImageWithFallback({ src, alt, onError, ...props }) {
  const [failed, setFailed] = React.useState(false);
  const handleError = (e) => {
    setFailed(true);
    onError?.();
  };
  if (failed) {
    return (
      <Box
        sx={{
          minHeight: 120,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'action.hover',
          borderRadius: 2,
          border: '1px dashed',
          borderColor: 'divider',
          color: 'text.secondary',
          fontSize: '0.875rem',
        }}
      >
        Image failed to load · try HTML view or Save HTML
      </Box>
    );
  }
  return (
    <img
      src={src}
      alt={alt || ''}
      {...props}
      style={{ maxWidth: '100%', height: 'auto', borderRadius: 8 }}
      onError={handleError}
    />
  );
}

const GraphicDocumentGenerator = () => {
  const [topic, setTopic] = useState('');
  const [llmProvider, setLlmProvider] = useState('gemini');
  const [modelName, setModelName] = useState('');
  const [maxImages, setMaxImages] = useState(3);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expandOpen, setExpandOpen] = useState(false);
  const [viewMode, setViewMode] = useState('markdown'); // 'markdown' | 'html'
  const [imageErrors, setImageErrors] = useState(0);

  const { data: providersData = { providers: [] } } = useQuery('providers', api.getProviders, { retry: false });
  const providers = (providersData && providersData.providers) ? providersData.providers : [];

  const { data: modelsData } = useQuery('models', api.getModels, { retry: false });
  const models = Array.isArray(modelsData) ? modelsData : [];
  const modelList = useMemo(() => {
    if (!llmProvider) return models.map((m) => m.name || m.id || m.model);
    return models
      .filter((m) => (m.provider || '').toLowerCase() === llmProvider.toLowerCase())
      .map((m) => m.name || m.id || m.model);
  }, [llmProvider, models]);

  useEffect(() => {
    if (modelList.length && !modelList.includes(modelName)) {
      setModelName(modelList[0] || '');
    }
  }, [llmProvider, modelList]);

  const generateMutation = useMutation(
    (payload) => api.generateGraphicDocument(payload),
    {
      onSuccess: (data) => {
        setResult(data);
        setError(data.error || null);
        setImageErrors(0);
        setViewMode(data.html_filename ? 'html' : 'markdown');
      },
      onError: (err) => {
        setError(err.response?.data?.detail || err.message || 'Generation failed');
        setResult(null);
      },
    }
  );

  const handleGenerate = () => {
    if (!topic.trim()) {
      setError('Please enter a topic.');
      return;
    }
    setError(null);
    generateMutation.mutate({
      topic: topic.trim(),
      llm_provider: llmProvider || 'gemini',
      model_name: modelName || undefined,
      max_images: maxImages,
    });
  };

  const markdownContent = result && result.success ? (result.markdown || '').trim() : '';
  const imagesGenerated = result && result.success ? (result.images_generated ?? 0) : 0;
  const htmlFilename = result && result.success ? result.html_filename : null;

  const markdownComponents = useMemo(
    () => ({
      img: ({ src, alt, ...props }) => {
        // Resolve /images/file/... against API (or use as-is when proxied)
        const href = src && src.startsWith('/') ? `${API_BASE}${src}` : src;
        return (
          <ImageWithFallback
            src={href}
            alt={alt || ''}
            onError={() => setImageErrors((n) => n + 1)}
            {...props}
          />
        );
      },
    }),
    []
  );

  const handleSaveMarkdown = () => {
    if (!markdownContent) return;
    const blob = new Blob([markdownContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `graphic-document-${topic.slice(0, 30).replace(/\s+/g, '-')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadHtml = () => {
    if (!htmlFilename) return;
    const url = `${API_BASE}/graphic-document/file/${encodeURIComponent(htmlFilename)}?download=1`;
    const a = document.createElement('a');
    a.href = url;
    a.download = htmlFilename;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.click();
  };

  return (
    <Box sx={{ py: 2 }}>
      <Typography variant="h4" sx={{ mb: 2 }}>
        Graphic Document Generator
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Enter a topic. AI will generate a detailed, creative markdown document and add up to 5 illustrations using the image generator.
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
                Settings
              </Typography>
              <TextField
                fullWidth
                label="Topic"
                placeholder="e.g. The Future of Renewable Energy"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                multiline
                rows={3}
                sx={{ mb: 2 }}
                InputProps={{ sx: { bgcolor: 'background.default' } }}
              />
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>AI Provider</InputLabel>
                <Select
                  value={llmProvider}
                  label="AI Provider"
                  onChange={(e) => setLlmProvider(e.target.value)}
                >
                  {providers.map((p) => (
                    <MenuItem key={p} value={p}>
                      {p}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Model</InputLabel>
                <Select
                  value={modelName}
                  label="Model"
                  onChange={(e) => setModelName(e.target.value)}
                >
                  {modelList.map((m) => (
                    <MenuItem key={m} value={m}>
                      {m}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Max images: {maxImages}
                </Typography>
                <Slider
                  value={maxImages}
                  onChange={(_, v) => setMaxImages(v)}
                  min={1}
                  max={5}
                  step={1}
                  valueLabelDisplay="auto"
                  marks={[1, 2, 3, 4, 5].map((n) => ({ value: n, label: n }))}
                />
              </Box>
              <Button
                fullWidth
                variant="contained"
                startIcon={generateMutation.isLoading ? <CircularProgress size={20} color="inherit" /> : <GenerateIcon />}
                onClick={handleGenerate}
                disabled={generateMutation.isLoading || !topic.trim()}
                size="large"
              >
                {generateMutation.isLoading ? 'Generating…' : 'Generate Document'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8} sx={{ display: 'flex', flexDirection: 'column', minHeight: 0, maxHeight: 'calc(100vh - 200px)' }}>
          <Card sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden', '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0, mb: 1, flexWrap: 'nowrap', gap: 1 }}>
                <Typography variant="h6" sx={{ flexShrink: 0 }}>Output</Typography>
                {markdownContent && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'nowrap', flexShrink: 0 }}>
                    {htmlFilename && (
                      <ButtonGroup size="small" variant="outlined" sx={{ '& .MuiButton-root': { fontSize: '0.75rem', py: 0.5, px: 1 } }}>
                        <Button
                          startIcon={<MarkdownIcon sx={{ fontSize: 16 }} />}
                          onClick={() => setViewMode('markdown')}
                          variant={viewMode === 'markdown' ? 'contained' : 'outlined'}
                        >
                          MD
                        </Button>
                        <Button
                          startIcon={<HtmlIcon sx={{ fontSize: 16 }} />}
                          onClick={() => setViewMode('html')}
                          variant={viewMode === 'html' ? 'contained' : 'outlined'}
                        >
                          HTML
                        </Button>
                      </ButtonGroup>
                    )}
                    {htmlFilename && (
                      <Button size="small" variant="outlined" onClick={handleDownloadHtml} sx={{ fontSize: '0.75rem', py: 0.5, px: 1 }}>
                        Save HTML
                      </Button>
                    )}
                    <Button size="small" variant="outlined" onClick={handleSaveMarkdown} sx={{ fontSize: '0.75rem', py: 0.5, px: 1 }}>
                      Save MD
                    </Button>
                    <Button
                      size="small"
                      variant="contained"
                      startIcon={<ExpandIcon />}
                      onClick={() => setExpandOpen(true)}
                      sx={{ fontSize: '0.75rem', py: 0.5, px: 1 }}
                    >
                      Expand
                    </Button>
                  </Box>
                )}
              </Box>
              {result && result.success && (imagesGenerated > 0 || imageErrors > 0) && (
                <Typography variant="caption" color="text.secondary" sx={{ mb: 1, flexShrink: 0 }}>
                  {imagesGenerated} image{imagesGenerated !== 1 ? 's' : ''} generated
                  {imageErrors > 0 && ` · ${imageErrors} failed to load (use HTML view)`}
                </Typography>
              )}
              {error && (
                <Alert severity="error" sx={{ mb: 2, flexShrink: 0 }} onClose={() => setError(null)}>
                  {error}
                </Alert>
              )}
              {!result && !generateMutation.isLoading && (
                <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary', flex: 1 }}>
                  <GenerateIcon sx={{ fontSize: 64, mb: 2, opacity: 0.3 }} />
                  <Typography>Enter a topic and click Generate Document.</Typography>
                </Box>
              )}
              {generateMutation.isLoading && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', py: 4, flex: 1 }}>
                  <CircularProgress />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    Writing content and generating images…
                  </Typography>
                </Box>
              )}
              {result && !generateMutation.isLoading && (
                <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden', pr: 0.5, display: 'flex', flexDirection: 'column' }}>
                  {result.success ? (
                    viewMode === 'html' && htmlFilename ? (
                      <Paper variant="outlined" sx={{ flex: 1, minHeight: 120, maxHeight: 220, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                        <iframe
                          title="Graphic document HTML preview"
                          src={`${API_BASE}/graphic-document/file/${encodeURIComponent(htmlFilename)}`}
                          style={{ flex: 1, width: '100%', minHeight: 120, maxHeight: 220, border: 'none' }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ p: 1, textAlign: 'center' }}>
                          Click Expand to view full document
                        </Typography>
                      </Paper>
                    ) : (
                      <Paper
                        variant="outlined"
                        sx={{
                          p: 2,
                          borderColor: 'primary.main',
                          bgcolor: 'rgba(25, 118, 210, 0.04)',
                          overflow: 'hidden',
                          maxHeight: 220,
                          display: 'flex',
                          flexDirection: 'column',
                        }}
                      >
                        <Box
                          className="markdown-body"
                          sx={{
                            overflow: 'auto',
                            maxHeight: 180,
                            wordBreak: 'break-word',
                            overflowWrap: 'break-word',
                            '& h1, & h2, & h3': { mt: 1.5, mb: 0.5, fontWeight: 600 },
                            '& p': { mb: 1, wordBreak: 'break-word', overflowWrap: 'break-word' },
                            '& ul, & ol': { pl: 2.5, mb: 1 },
                            '& pre': { overflow: 'auto', p: 1.5, bgcolor: 'action.hover', borderRadius: 1, whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
                            '& code': { fontFamily: 'monospace', fontSize: '0.9em', wordBreak: 'break-word', overflowWrap: 'break-word' },
                            '& img': { maxWidth: '100%', height: 'auto', borderRadius: 1 },
                          }}
                        >
                          <ReactMarkdown components={markdownComponents}>{markdownContent || 'No content.'}</ReactMarkdown>
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, flexShrink: 0 }}>
                          Click Expand to view full document with images
                        </Typography>
                      </Paper>
                    )
                  ) : (
                    <Alert severity="error">{result.error || 'Generation failed.'}</Alert>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Dialog
        open={expandOpen}
        onClose={() => setExpandOpen(false)}
        maxWidth={false}
        fullWidth
        sx={{
          '& .MuiDialog-paper': {
            width: '90vw',
            maxWidth: 960,
            height: '85vh',
            maxHeight: 720,
          },
        }}
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', py: 1.5 }}>
          <Typography variant="h6">Document</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {markdownContent && htmlFilename && (
              <ButtonGroup size="small" variant="outlined">
                <Button
                  startIcon={<MarkdownIcon sx={{ fontSize: 16 }} />}
                  onClick={() => setViewMode('markdown')}
                  variant={viewMode === 'markdown' ? 'contained' : 'outlined'}
                >
                  MD
                </Button>
                <Button
                  startIcon={<HtmlIcon sx={{ fontSize: 16 }} />}
                  onClick={() => setViewMode('html')}
                  variant={viewMode === 'html' ? 'contained' : 'outlined'}
                >
                  HTML
                </Button>
              </ButtonGroup>
            )}
            <IconButton size="small" onClick={() => setExpandOpen(false)} aria-label="Close">
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent dividers sx={{ p: 0, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
          {markdownContent && (
            viewMode === 'html' && htmlFilename ? (
              <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
                <iframe
                  title="Graphic document HTML full"
                  src={`${API_BASE}/graphic-document/file/${encodeURIComponent(htmlFilename)}`}
                  style={{ width: '100%', height: '100%', minHeight: 400, border: 'none' }}
                />
              </Box>
            ) : (
              <Box
                sx={{
                  p: 3,
                  overflow: 'auto',
                  flex: 1,
                  wordBreak: 'break-word',
                  overflowWrap: 'break-word',
                  '& h1': { fontSize: '1.75rem', fontWeight: 700, mt: 0, mb: 1 },
                  '& h2': { fontSize: '1.35rem', fontWeight: 600, mt: 2, mb: 0.75 },
                  '& h3': { fontSize: '1.1rem', fontWeight: 600, mt: 1.5, mb: 0.5 },
                  '& p': { mb: 1.25, lineHeight: 1.7 },
                  '& ul, & ol': { pl: 2.5, mb: 1.25 },
                  '& img': { maxWidth: '100%', height: 'auto', borderRadius: 2, my: 1.5 },
                  '& pre': { overflow: 'auto', p: 2, bgcolor: 'action.hover', borderRadius: 1 },
                  '& code': { fontFamily: 'monospace', fontSize: '0.9em' },
                }}
              >
                <ReactMarkdown components={markdownComponents}>{markdownContent}</ReactMarkdown>
              </Box>
            )
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default GraphicDocumentGenerator;
