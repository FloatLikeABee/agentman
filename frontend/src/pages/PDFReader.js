import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Grid,
  CircularProgress,
  Alert,
  Paper,
  Chip,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  PictureAsPdf as PDFIcon,
  Upload as UploadIcon,
  Delete as DeleteIcon,
  SmartToy as AIIcon,
  LibraryAdd as RAGIcon,
  AutoAwesome as SuggestIcon,
  Save as SaveIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import ReactMarkdown from 'react-markdown';
import api from '../services/api';
import { useQuery } from 'react-query';

const PDFReader = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [llmProvider, setLlmProvider] = useState('');
  const [modelName, setModelName] = useState('');
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [ragModalOpen, setRagModalOpen] = useState(false);
  const [ragCollectionName, setRagCollectionName] = useState('');
  const [ragTitle, setRagTitle] = useState('');
  const [ragDescription, setRagDescription] = useState('');
  const [ragSuggestingTitle, setRagSuggestingTitle] = useState(false);
  const [ragAdding, setRagAdding] = useState(false);
  const [ragError, setRagError] = useState(null);

  // Fetch providers and models
  const { data: providersData = { providers: [] } } = useQuery('providers', api.getProviders);
  const providers = providersData.providers || [];
  
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

  // Set default provider and model
  React.useEffect(() => {
    if (providers.length > 0 && !llmProvider) {
      setLlmProvider(providers[0]);
    }
  }, [providers, llmProvider]);

  React.useEffect(() => {
    if (models[llmProvider] && models[llmProvider].length > 0 && !modelName) {
      setModelName(models[llmProvider][0]);
    }
  }, [llmProvider, models, modelName]);

  const { data: ragCollectionsData = [] } = useQuery('rag-collections', api.getRAGCollections, { retry: false });
  const ragCollections = Array.isArray(ragCollectionsData) ? ragCollectionsData.map((c) => (typeof c === 'string' ? c : c.name)).filter(Boolean) : [];

  const getContentForRAG = () => {
    if (!result) return '';
    return (result.ai_result || result.extracted_text || '').trim();
  };

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
          tags: ['pdf-reader'],
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
    if (!result) return '';
    const meta = [];
    if (result.page_count !== undefined) meta.push(`${result.page_count} pages`);
    if (result.extracted_text_length) meta.push(`${result.extracted_text_length.toLocaleString()} characters`);
    if (result.provider) meta.push(`${result.provider} / ${result.model || 'default'}`);
    if (result.timestamp) meta.push(`Processed: ${new Date(result.timestamp).toLocaleString()}`);
    let md = '# PDF Read Result\n\n';
    if (meta.length) md += meta.join(' · ') + '\n\n';
    if (result.ai_result) md += '## AI Result\n\n' + result.ai_result + '\n\n';
    if (result.extracted_text) {
      const text = result.extracted_text.length > 1000
        ? result.extracted_text.substring(0, 1000) + '\n\n[Truncated. Full text has ' + result.extracted_text.length.toLocaleString() + ' characters.]'
        : result.extracted_text;
      md += '## Extracted Text\n\n```\n' + text + '\n```\n';
    }
    return md;
  };

  const handleSaveToFile = () => {
    const md = getMarkdownForSave();
    if (!md) return;
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pdf-read-${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '-')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedFile({
        file,
        name: file.name,
        size: file.size,
      });
      setError(null);
      setResult(null);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
  });

  const removeFile = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  const handleReadPDF = async () => {
    if (!selectedFile) {
      setError('Please select a PDF file');
      return;
    }

    if (!systemPrompt || !systemPrompt.trim()) {
      setError('Please enter a system prompt');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile.file);
      formData.append('system_prompt', systemPrompt);
      if (llmProvider) formData.append('llm_provider', llmProvider);
      if (modelName) formData.append('model_name', modelName);

      const response = await api.readPDF(formData);
      setResult(response);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to read PDF');
    } finally {
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">PDF Reader</Typography>
        <Typography variant="body2" color="text.secondary">
          Extract text from PDF and process with AI
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ alignItems: 'stretch' }}>
        {/* Upload and Configuration Section */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload PDF
              </Typography>

              {/* Dropzone */}
              <Box
                {...getRootProps()}
                sx={{
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  borderRadius: 2,
                  p: 3,
                  textAlign: 'center',
                  cursor: 'pointer',
                  bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                  mb: 2,
                  transition: 'all 0.2s',
                  '&:hover': {
                    borderColor: 'primary.main',
                    bgcolor: 'action.hover',
                  },
                }}
              >
                <input {...getInputProps()} />
                <PDFIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
                <Typography variant="body1" gutterBottom>
                  {isDragActive
                    ? 'Drop PDF here'
                    : 'Drag & drop PDF here, or click to select'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Supports PDF files
                </Typography>
              </Box>

              {/* Selected File */}
              {selectedFile && (
                <Paper
                  sx={{
                    p: 2,
                    mb: 2,
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {selectedFile.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatFileSize(selectedFile.size)}
                    </Typography>
                  </Box>
                  <Button
                    size="small"
                    onClick={removeFile}
                    color="error"
                    startIcon={<DeleteIcon />}
                  >
                    Remove
                  </Button>
                </Paper>
              )}

              {/* System Prompt */}
              <TextField
                fullWidth
                label="System Prompt"
                placeholder="e.g., Summarize the key points from this document"
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                multiline
                rows={4}
                required
                sx={{ mb: 2 }}
                helperText="Enter instructions for how the AI should process the PDF content"
              />

              {/* LLM Provider Selection */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>LLM Provider (Optional)</InputLabel>
                <Select
                  value={llmProvider}
                  onChange={(e) => {
                    setLlmProvider(e.target.value);
                    setModelName(''); // Reset model when provider changes
                  }}
                  label="LLM Provider (Optional)"
                >
                  {providers.map((provider) => (
                    <MenuItem key={provider} value={provider}>
                      {provider.charAt(0).toUpperCase() + provider.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Model Selection */}
              {llmProvider && models[llmProvider] && models[llmProvider].length > 0 && (
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Model (Optional)</InputLabel>
                  <Select
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    label="Model (Optional)"
                  >
                    {models[llmProvider].map((model) => (
                      <MenuItem key={model} value={model}>
                        {model}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}

              {/* Read Button */}
              <Button
                fullWidth
                variant="contained"
                startIcon={<AIIcon />}
                onClick={handleReadPDF}
                disabled={!selectedFile || !systemPrompt.trim() || isProcessing}
                size="large"
              >
                {isProcessing ? 'Processing...' : 'Read & Process PDF'}
              </Button>

              {/* Processing Progress */}
              {isProcessing && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Extracting text and processing with AI...
                  </Typography>
                  <LinearProgress sx={{ mt: 1 }} />
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column', minHeight: 0, maxHeight: 'calc(100vh - 180px)' }}>
          <Card sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden', '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0, mb: 1, flexWrap: 'wrap', gap: 1 }}>
                <Typography variant="h6">
                  Results
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexShrink: 0 }}>
                  {result && getMarkdownForSave() && (
                    <Button size="small" variant="outlined" startIcon={<SaveIcon />} onClick={handleSaveToFile}>
                      Save to file
                    </Button>
                  )}
                  {result && getContentForRAG() && (
                    <Button size="small" variant="outlined" startIcon={<RAGIcon />} onClick={handleOpenRagModal}>
                      Add to RAG
                    </Button>
                  )}
                </Box>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mb: 2, flexShrink: 0 }} onClose={() => setError(null)}>
                  {error}
                </Alert>
              )}

              {isProcessing && (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4, flex: 1 }}>
                  <CircularProgress />
                </Box>
              )}

              {!isProcessing && !result && (
                <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary', flex: 1 }}>
                  <PDFIcon sx={{ fontSize: 64, mb: 2, opacity: 0.3 }} />
                  <Typography>No results yet. Upload a PDF and process it to see results.</Typography>
                </Box>
              )}

              {!isProcessing && result && (
                <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto', pr: 0.5 }}>
                  {/* Success Indicator */}
                  {result.success && (
                    <Alert severity="success" sx={{ mb: 2 }}>
                      PDF processed successfully!
                    </Alert>
                  )}

                  {/* Metadata */}
                  {result.page_count !== undefined && (
                    <Box sx={{ mb: 2 }}>
                      <Chip
                        label={`${result.page_count} pages`}
                        size="small"
                        sx={{ mr: 1 }}
                      />
                      {result.extracted_text_length && (
                        <Chip
                          label={`${result.extracted_text_length.toLocaleString()} characters`}
                          size="small"
                          sx={{ mr: 1 }}
                        />
                      )}
                      {result.provider && (
                        <Chip
                          label={`${result.provider} / ${result.model || 'default'}`}
                          size="small"
                          color="primary"
                        />
                      )}
                    </Box>
                  )}

                  {/* AI Result (Markdown) */}
                  {result.ai_result && (
                    <Paper
                      sx={{
                        p: 2,
                        mb: 2,
                        border: '1px solid',
                        borderColor: 'primary.main',
                        borderRadius: 1,
                        bgcolor: 'rgba(25, 118, 210, 0.05)',
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <AIIcon sx={{ mr: 1, color: 'primary.main' }} />
                        <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                          AI Processed Result
                        </Typography>
                      </Box>
                      <Divider sx={{ my: 1 }} />
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
                        <ReactMarkdown>{result.ai_result}</ReactMarkdown>
                      </Box>
                    </Paper>
                  )}

                  {/* Extracted Text Preview */}
                  {result.extracted_text && (
                    <Paper
                      sx={{
                        p: 2,
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                      }}
                    >
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                        Extracted Text (Preview)
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{
                          whiteSpace: 'pre-wrap',
                          fontFamily: 'monospace',
                          bgcolor: 'background.default',
                          p: 1,
                          borderRadius: 1,
                          maxHeight: 'min(280px, 30vh)',
                          overflow: 'auto',
                          fontSize: '0.85rem',
                        }}
                      >
                        {result.extracted_text.length > 1000
                          ? `${result.extracted_text.substring(0, 1000)}...\n\n[Text truncated. Full text has ${result.extracted_text.length.toLocaleString()} characters.]`
                          : result.extracted_text}
                      </Typography>
                    </Paper>
                  )}

                  {/* System Prompt Used */}
                  {result.system_prompt && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="caption" color="text.secondary">
                        System Prompt: {result.system_prompt}
                      </Typography>
                    </Box>
                  )}

                  {/* Timestamp */}
                  {result.timestamp && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      Processed: {new Date(result.timestamp).toLocaleString()}
                    </Typography>
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
            Save the current result to a RAG collection for later retrieval. A topic title will be suggested from the content.
          </Typography>
          {ragError && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setRagError(null)}>
              {ragError}
            </Alert>
          )}
          <TextField
            fullWidth
            label="Collection name"
            placeholder="e.g. my_notes"
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
              placeholder="Topic title for this content"
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

export default PDFReader;
