import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Alert,
  CircularProgress,
  Grid,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Search as SearchIcon,
  Save as SaveIcon,
  LibraryAdd as RAGIcon,
  AutoAwesome as SuggestIcon,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { useQuery, useMutation } from 'react-query';
import api from '../services/api';

const GATHERING_SYSTEM_PROMPT_PREVIEW = `You are a research assistant. Gather information using:
1. Wikipedia (factual overview)
2. Reddit (site:reddit.com via web search)
3. General web search

Synthesize into a structured markdown report.`;

const Gathering = () => {
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [maxIterations, setMaxIterations] = useState(10);
  const [llmProvider, setLlmProvider] = useState('');
  const [modelName, setModelName] = useState('');
  const [ragModalOpen, setRagModalOpen] = useState(false);
  const [ragCollectionName, setRagCollectionName] = useState('');
  const [ragTitle, setRagTitle] = useState('');
  const [ragDescription, setRagDescription] = useState('');
  const [ragSuggestingTitle, setRagSuggestingTitle] = useState(false);
  const [ragAdding, setRagAdding] = useState(false);
  const [ragError, setRagError] = useState(null);

  const { data: providersData = { providers: [] } } = useQuery('providers', api.getProviders, { retry: false });
  const providers = (providersData && providersData.providers) ? providersData.providers : [];

  const { data: modelsData } = useQuery('models', api.getModels, { retry: false });
  const models = Array.isArray(modelsData) ? modelsData : [];
  const modelList = llmProvider
    ? models.filter((m) => (m.provider || '').toLowerCase() === llmProvider.toLowerCase()).map((m) => m.name || m.id || m.model)
    : models.map((m) => m.name || m.id || m.model);

  const { data: ragCollectionsData = [] } = useQuery('rag-collections', api.getRAGCollections, { retry: false });
  const ragCollections = Array.isArray(ragCollectionsData)
    ? ragCollectionsData.map((c) => (typeof c === 'string' ? c : c.name)).filter(Boolean)
    : [];

  const gatherMutation = useMutation(
    (payload) => api.gatherData(payload),
    {
      onSuccess: (data) => {
        setResult(data);
        setError(null);
      },
      onError: (err) => {
        setError(err.response?.data?.detail || err.message || 'Gathering failed');
        setResult(null);
      },
    }
  );

  const handleGather = () => {
    if (!prompt.trim()) {
      setError('Please enter a topic or question to research.');
      return;
    }
    setError(null);
    gatherMutation.mutate({
      prompt: prompt.trim(),
      max_iterations: maxIterations,
      llm_provider: llmProvider || undefined,
      model_name: modelName || undefined,
    });
  };

  const getContentForRAG = () => (result && result.success ? (result.content || '').trim() : '');

  const handleOpenRagModal = async () => {
    const content = getContentForRAG();
    if (!content) return;
    setRagModalOpen(true);
    setRagError(null);
    setRagTitle('');
    setRagDescription('');
    setRagCollectionName(ragCollections[0] || '');
    setRagSuggestingTitle(true);
    try {
      const res = await api.suggestRAGTitle(content);
      setRagTitle((res && res.title) ? res.title : '');
    } catch {
      setRagTitle('');
    } finally {
      setRagSuggestingTitle(false);
    }
  };

  const handleAddToRAG = async () => {
    const collection = (ragCollectionName || '').trim();
    const title = (ragTitle || '').trim();
    const content = getContentForRAG();
    if (!collection || !title || !content) {
      setRagError('Collection name and document title are required.');
      return;
    }
    setRagError(null);
    setRagAdding(true);
    try {
      await api.addRAGData({
        collection_name: collection,
        data_input: {
          name: title,
          description: ragDescription.trim() || undefined,
          format: 'txt',
          content,
          tags: ['gathering'],
        },
      });
      setRagModalOpen(false);
    } catch (err) {
      setRagError(err.response?.data?.detail || err.message || 'Failed to add to RAG');
    } finally {
      setRagAdding(false);
    }
  };

  const getMarkdownForSave = () => {
    if (!result || !result.success) return '';
    const meta = [];
    if (result.provider) meta.push(`Provider: ${result.provider}`);
    if (result.model) meta.push(`Model: ${result.model}`);
    if (result.max_iterations) meta.push(`Max iterations: ${result.max_iterations}`);
    let md = '# Gathered Research\n\n';
    if (meta.length) md += meta.join(' · ') + '\n\n';
    md += '---\n\n';
    md += result.content || '';
    return md;
  };

  const handleSaveToFile = () => {
    const md = getMarkdownForSave();
    if (!md) return;
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gathering-${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '-')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Gathering</Typography>
        <Typography variant="body2" color="text.secondary">
          AI-powered research from Wikipedia, Reddit, and web search
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ alignItems: 'stretch' }}>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Research Topic
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Enter a topic or question. The AI will gather information from Wikipedia, Reddit (via web search), and general web search, then synthesize a markdown report.
              </Typography>

              <TextField
                fullWidth
                multiline
                rows={4}
                label="Topic or question"
                placeholder="e.g. Best practices for learning Python in 2024"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                sx={{ mb: 2 }}
                InputProps={{ sx: { bgcolor: 'background.default' } }}
              />

              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
                <FormControl size="small" sx={{ minWidth: 140 }}>
                  <InputLabel>Max iterations</InputLabel>
                  <Select
                    value={maxIterations}
                    label="Max iterations"
                    onChange={(e) => setMaxIterations(Number(e.target.value))}
                  >
                    {[5, 8, 10, 12, 15, 20].map((n) => (
                      <MenuItem key={n} value={n}>{n}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Provider</InputLabel>
                  <Select
                    value={llmProvider}
                    label="Provider"
                    onChange={(e) => {
                      setLlmProvider(e.target.value);
                      setModelName('');
                    }}
                  >
                    <MenuItem value="">Default</MenuItem>
                    {providers.map((p) => (
                      <MenuItem key={p} value={p}>{p}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl size="small" sx={{ minWidth: 160 }}>
                  <InputLabel>Model</InputLabel>
                  <Select
                    value={modelName}
                    label="Model"
                    onChange={(e) => setModelName(e.target.value)}
                  >
                    <MenuItem value="">Default</MenuItem>
                    {modelList.map((m) => (
                      <MenuItem key={m} value={m}>{m}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>

              <Paper variant="outlined" sx={{ p: 1.5, mb: 2, bgcolor: 'action.hover' }}>
                <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                  Preset system prompt (used by AI):
                </Typography>
                <Typography variant="body2" sx={{ mt: 0.5, fontFamily: 'monospace', fontSize: '0.75rem', whiteSpace: 'pre-wrap' }}>
                  {GATHERING_SYSTEM_PROMPT_PREVIEW}
                </Typography>
              </Paper>

              <Button
                fullWidth
                variant="contained"
                startIcon={gatherMutation.isLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
                onClick={handleGather}
                disabled={gatherMutation.isLoading || !prompt.trim()}
                size="large"
              >
                {gatherMutation.isLoading ? 'Gathering…' : 'Gather'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column', minHeight: 0, maxHeight: 'calc(100vh - 180px)' }}>
          <Card sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden', '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0, mb: 1, flexWrap: 'wrap', gap: 1 }}>
                <Typography variant="h6">Results</Typography>
                <Box sx={{ display: 'flex', gap: 1, flexShrink: 0 }}>
                  {result && result.success && getMarkdownForSave() && (
                    <>
                      <Button size="small" variant="outlined" startIcon={<SaveIcon />} onClick={handleSaveToFile}>
                        Save to Markdown
                      </Button>
                      <Button size="small" variant="outlined" startIcon={<RAGIcon />} onClick={handleOpenRagModal}>
                        Add to RAG
                      </Button>
                    </>
                  )}
                </Box>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mb: 2, flexShrink: 0 }} onClose={() => setError(null)}>
                  {error}
                </Alert>
              )}

              {!result && !gatherMutation.isLoading && (
                <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary', flex: 1 }}>
                  <SearchIcon sx={{ fontSize: 64, mb: 2, opacity: 0.3 }} />
                  <Typography>No results yet. Enter a topic and click Gather.</Typography>
                </Box>
              )}

              {gatherMutation.isLoading && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', py: 4, flex: 1 }}>
                  <CircularProgress />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    Searching Wikipedia, Reddit, and web…
                  </Typography>
                </Box>
              )}

              {result && !gatherMutation.isLoading && (
                <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto', pr: 0.5 }}>
                  {result.success ? (
                    <Paper
                      sx={{
                        p: 2,
                        border: '1px solid',
                        borderColor: 'primary.main',
                        borderRadius: 1,
                        bgcolor: 'rgba(25, 118, 210, 0.05)',
                      }}
                    >
                      <Box
                        className="markdown-body"
                        sx={{
                          '& h1, & h2, & h3': { mt: 1.5, mb: 0.5, fontWeight: 600 },
                          '& p': { mb: 1 },
                          '& ul, & ol': { pl: 2.5, mb: 1 },
                          '& pre': { overflow: 'auto', p: 1.5, bgcolor: 'action.hover', borderRadius: 1 },
                          '& code': { fontFamily: 'monospace', fontSize: '0.9em' },
                        }}
                      >
                        <ReactMarkdown>{result.content || 'No content.'}</ReactMarkdown>
                      </Box>
                    </Paper>
                  ) : (
                    <Alert severity="error">{result.error || 'Gathering failed.'}</Alert>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Add to RAG modal */}
      <Dialog open={ragModalOpen} onClose={() => !ragAdding && setRagModalOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add to RAG</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Save the gathered content to a RAG collection for later retrieval.
          </Typography>
          {ragError && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setRagError(null)}>
              {ragError}
            </Alert>
          )}
          <TextField
            fullWidth
            label="Collection name"
            placeholder="e.g. my_research"
            value={ragCollectionName}
            onChange={(e) => setRagCollectionName(e.target.value)}
            sx={{ mb: 2 }}
            helperText={ragCollections.length ? 'Or choose existing below' : 'Collection will be created if new'}
            InputProps={{ sx: { bgcolor: 'background.default' } }}
          />
          {ragCollections.length > 0 && (
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Existing collections</InputLabel>
              <Select
                value={ragCollectionName}
                label="Existing collections"
                onChange={(e) => setRagCollectionName(e.target.value)}
              >
                {ragCollections.map((c) => (
                  <MenuItem key={c} value={c}>{c}</MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-start', mb: 2 }}>
            <TextField
              fullWidth
              label="Document title"
              placeholder="Topic title"
              value={ragTitle}
              onChange={(e) => setRagTitle(e.target.value)}
              disabled={ragSuggestingTitle}
              InputProps={{ sx: { bgcolor: 'background.default' } }}
            />
            <Button
              variant="outlined"
              size="small"
              onClick={async () => {
                const content = getContentForRAG();
                if (!content) return;
                setRagSuggestingTitle(true);
                try {
                  const res = await api.suggestRAGTitle(content);
                  setRagTitle((res && res.title) ? res.title : '');
                } finally {
                  setRagSuggestingTitle(false);
                }
              }}
              disabled={ragSuggestingTitle || !getContentForRAG()}
              sx={{ minWidth: 48, mt: 1 }}
              title="Suggest title with AI"
            >
              {ragSuggestingTitle ? <CircularProgress size={24} /> : <SuggestIcon />}
            </Button>
          </Box>
          <TextField
            fullWidth
            label="Description (optional)"
            placeholder="Brief description"
            value={ragDescription}
            onChange={(e) => setRagDescription(e.target.value)}
            multiline
            rows={2}
            InputProps={{ sx: { bgcolor: 'background.default' } }}
          />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setRagModalOpen(false)} disabled={ragAdding}>Cancel</Button>
          <Button variant="contained" onClick={handleAddToRAG} disabled={ragAdding || !ragTitle.trim() || !ragCollectionName.trim()} startIcon={ragAdding ? <CircularProgress size={18} color="inherit" /> : <RAGIcon />}>
            {ragAdding ? 'Adding…' : 'Add to RAG'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Gathering;
