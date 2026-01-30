import React, { useState, useEffect, useMemo } from 'react';
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
  LinearProgress,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Image as ImageIcon,
  Upload as UploadIcon,
  Delete as DeleteIcon,
  SmartToy as AIIcon,
  LibraryAdd as RAGIcon,
  AutoAwesome as SuggestIcon,
  Save as SaveIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useQuery } from 'react-query';
import ReactMarkdown from 'react-markdown';
import api from '../services/api';

const ImageReader = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [results, setResults] = useState([]);
  const [processResult, setProcessResult] = useState(null); // Read + AI result
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [totalImages, setTotalImages] = useState(0);
  const [systemPrompt, setSystemPrompt] = useState('Summarize the following content clearly. Extract key points and answer any implied questions.');
  const [selectedProvider, setSelectedProvider] = useState('qwen');
  const [selectedModel, setSelectedModel] = useState('');
  const [resultTab, setResultTab] = useState(0); // 0 = extracted text, 1 = AI result
  const [ragModalOpen, setRagModalOpen] = useState(false);
  const [ragCollectionName, setRagCollectionName] = useState('');
  const [ragTitle, setRagTitle] = useState('');
  const [ragDescription, setRagDescription] = useState('');
  const [ragSuggestingTitle, setRagSuggestingTitle] = useState(false);
  const [ragAdding, setRagAdding] = useState(false);
  const [ragError, setRagError] = useState(null);

  const { data: providersData, isLoading: providersLoading, isError: providersError } = useQuery(
    'providers',
    api.getProviders,
    { retry: false }
  );
  const providers = (providersData && providersData.providers) ? providersData.providers : [];

  const { data: modelsData, isLoading: modelsLoading, isError: modelsError } = useQuery(
    'models',
    api.getModels,
    { retry: false }
  );
  const models = useMemo(() => {
    const byProvider = {};
    const list = Array.isArray(modelsData)
      ? modelsData
      : (modelsData?.models && Array.isArray(modelsData.models) ? modelsData.models : []);
    list.forEach((m) => {
      const p = (m && m.provider) ? String(m.provider).toLowerCase() : '';
      if (!p) return;
      if (!byProvider[p]) byProvider[p] = [];
      const name = m.name || m.id || m.model;
      if (name && !byProvider[p].includes(name)) byProvider[p].push(name);
    });
    return byProvider;
  }, [modelsData]);

  useEffect(() => {
    if (providers.length && !providers.includes(selectedProvider)) {
      setSelectedProvider(providers[0]);
    }
  }, [providers, selectedProvider]);

  const providerKey = selectedProvider ? String(selectedProvider).toLowerCase() : '';
  const modelList = models[providerKey] || [];
  const defaultModelByProvider = { qwen: 'qwen-vl-plus', gemini: 'gemini-1.5-flash', mistral: 'mistral-small' };
  const effectiveModel = selectedModel || (modelList && modelList[0]) || (providerKey ? defaultModelByProvider[providerKey] : null);

  const { data: ragCollectionsData = [] } = useQuery('rag-collections', api.getRAGCollections, { retry: false });
  const ragCollections = Array.isArray(ragCollectionsData) ? ragCollectionsData.map((c) => (typeof c === 'string' ? c : c.name)).filter(Boolean) : [];

  const getContentForRAG = () => {
    if (processResult) return (processResult.ai_result || processResult.extracted_text || '').trim();
    if (results.length > 0) return results.map((r) => r.text).filter(Boolean).join('\n\n').trim();
    return '';
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
          tags: ['image-reader'],
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
    if (processResult) {
      const meta = [];
      if (processResult.provider) meta.push(`Provider: ${processResult.provider}`);
      if (processResult.model) meta.push(`Model: ${processResult.model}`);
      if (processResult.timestamp) meta.push(`Processed: ${new Date(processResult.timestamp).toLocaleString()}`);
      let md = '# Image Read & AI Result\n\n';
      if (meta.length) md += meta.join(' · ') + '\n\n';
      md += '## AI Result\n\n' + (processResult.ai_result || 'No AI result') + '\n\n';
      if (processResult.extracted_text) md += '## Extracted Text\n\n```\n' + processResult.extracted_text + '\n```\n';
      return md;
    }
    if (results.length > 0) {
      let md = '# Image OCR Results\n\n';
      results.forEach((r, i) => {
        md += '## ' + (results.length > 1 ? `Image ${r.image_index || i + 1}` : 'Result') + '\n\n';
        if (r.success && r.text) md += '```\n' + r.text + '\n```\n\n';
        else if (r.error) md += '*Error: ' + r.error + '*\n\n';
      });
      return md;
    }
    return '';
  };

  const handleSaveToFile = () => {
    const md = getMarkdownForSave();
    if (!md) return;
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `image-read-${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '-')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    const list = models[providerKey] || [];
    if (Array.isArray(list) && list.length) {
      if (!list.includes(selectedModel)) {
        setSelectedModel(list[0]);
      }
    } else {
      setSelectedModel('');
    }
  }, [selectedProvider, models, providerKey, selectedModel]);

  const onDrop = (acceptedFiles) => {
    // Limit to 5 images
    const filesToAdd = acceptedFiles.slice(0, 5 - selectedFiles.length);
    const newFiles = filesToAdd.map((file) => ({
      file,
      preview: URL.createObjectURL(file),
      name: file.name,
    }));
    setSelectedFiles((prev) => [...prev, ...newFiles]);
    setError(null);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp', '.bmp'],
    },
    maxFiles: 5,
    disabled: selectedFiles.length >= 5,
  });

  const removeFile = (index) => {
    const newFiles = [...selectedFiles];
    URL.revokeObjectURL(newFiles[index].preview);
    newFiles.splice(index, 1);
    setSelectedFiles(newFiles);
  };

  const clearAll = () => {
    selectedFiles.forEach((file) => URL.revokeObjectURL(file.preview));
    setSelectedFiles([]);
    setResults([]);
    setProcessResult(null);
    setError(null);
    setCurrentImageIndex(0);
    setTotalImages(0);
  };

  const handleReadAndProcess = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image');
      return;
    }
    if (!systemPrompt?.trim()) {
      setError('Please enter a system prompt for the AI');
      return;
    }
    const modelToUse = selectedModel || (modelList && modelList[0]) || (providerKey ? defaultModelByProvider[providerKey] : null);
    if (!modelToUse) {
      setError('Please select an AI model (or ensure the backend is running to load models).');
      return;
    }
    setIsProcessing(true);
    setError(null);
    setResults([]);
    setProcessResult(null);
    setTotalImages(selectedFiles.length);
    try {
      const data = selectedFiles.length === 1
        ? await api.readImageAndProcess(
            selectedFiles[0].file,
            systemPrompt,
            selectedProvider,
            modelToUse,
            prompt || null
          )
        : await api.readImageAndProcessMultiple(
            selectedFiles.map((f) => f.file),
            systemPrompt,
            selectedProvider,
            modelToUse,
            prompt || null
          );
      setProcessResult(data);
      setResultTab(1); // show AI result first
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Read & process failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReadImages = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResults([]);
    setProcessResult(null);
    setCurrentImageIndex(0);
    setTotalImages(selectedFiles.length);

    try {
      const files = selectedFiles.map((item) => item.file);
      
      if (files.length === 1) {
        // Single image
        const result = await api.readImage(files[0], prompt || null);
        if (result.success) {
          setResults([result]);
        } else {
          setError(result.error || 'Failed to read image');
        }
      } else {
        // Multiple images - process one by one
        const allResults = [];
        for (let i = 0; i < files.length; i++) {
          setCurrentImageIndex(i + 1);
          const result = await api.readImage(files[i], prompt || null);
          if (result.success) {
            allResults.push({ ...result, image_index: i + 1 });
          } else {
            allResults.push({
              success: false,
              error: result.error || 'Failed to read image',
              image_index: i + 1,
            });
          }
        }
        setResults(allResults);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to read images');
    } finally {
      setIsProcessing(false);
      setCurrentImageIndex(0);
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Image Reader</Typography>
        <Typography variant="body2" color="text.secondary">
          Extract text from images using Qwen Vision OCR
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ alignItems: 'stretch' }}>
        {/* Upload Section */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload Images (1-5 images)
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
                <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
                <Typography variant="body1" gutterBottom>
                  {isDragActive
                    ? 'Drop images here'
                    : selectedFiles.length >= 5
                    ? 'Maximum 5 images reached'
                    : 'Drag & drop images here, or click to select'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Supports JPEG, PNG, GIF, WebP, BMP (Max 5 images)
                </Typography>
              </Box>

              {/* Selected Files Preview */}
              {selectedFiles.length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle2">
                      Selected Images ({selectedFiles.length}/5)
                    </Typography>
                    <Button size="small" onClick={clearAll} color="error">
                      Clear All
                    </Button>
                  </Box>
                  <Grid container spacing={1}>
                    {selectedFiles.map((item, index) => (
                      <Grid item xs={6} sm={4} key={index}>
                        <Paper
                          sx={{
                            position: 'relative',
                            p: 0.5,
                            border: '1px solid',
                            borderColor: 'divider',
                            borderRadius: 1,
                          }}
                        >
                          <Box
                            sx={{
                              width: '100%',
                              paddingTop: '75%',
                              position: 'relative',
                              overflow: 'hidden',
                              borderRadius: 0.5,
                            }}
                          >
                            <img
                              src={item.preview}
                              alt={item.name}
                              style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                              }}
                            />
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 0.5 }}>
                            <Typography
                              variant="caption"
                              sx={{
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                                flex: 1,
                              }}
                            >
                              {item.name}
                            </Typography>
                            <IconButton
                              size="small"
                              onClick={() => removeFile(index)}
                              sx={{ p: 0.5 }}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}

              {/* Custom OCR Prompt */}
              <TextField
                fullWidth
                label="Custom OCR Prompt (Optional)"
                placeholder="Leave empty to use default: extract text only"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                multiline
                rows={2}
                sx={{ mb: 2 }}
              />

              {/* System prompt + AI model for Read & Process */}
              <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>Process with AI (optional)</Typography>
              <TextField
                fullWidth
                label="System prompt for AI"
                placeholder="e.g. Summarize the following content. Extract key points."
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                multiline
                rows={3}
                sx={{ mb: 1.5 }}
              />
              {(providersError || modelsError) && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  Cannot load AI providers/models — backend not reachable. Start the backend in a separate terminal: <strong>python main.py</strong> (from project root). Then refresh this page.
                </Alert>
              )}
              <Grid container spacing={1} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <FormControl fullWidth size="small" disabled={providersLoading}>
                    <InputLabel id="image-reader-provider-label">AI Provider</InputLabel>
                    <Select
                      labelId="image-reader-provider-label"
                      value={providers.length ? selectedProvider : ''}
                      label="AI Provider"
                      onChange={(e) => setSelectedProvider(e.target.value)}
                      displayEmpty
                      renderValue={(v) => v ? String(v).charAt(0).toUpperCase() + String(v).slice(1) : (providersLoading ? 'Loading...' : 'No providers')}
                    >
                      <MenuItem value="" disabled>No providers (start backend)</MenuItem>
                      {providers.map((p) => (
                        <MenuItem key={p} value={p}>{String(p).charAt(0).toUpperCase() + String(p).slice(1)}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6}>
                  <FormControl fullWidth size="small" disabled={modelsLoading}>
                    <InputLabel id="image-reader-model-label">Model</InputLabel>
                    <Select
                      labelId="image-reader-model-label"
                      value={selectedModel || (modelList[0] || '')}
                      label="Model"
                      onChange={(e) => setSelectedModel(e.target.value)}
                      displayEmpty
                      renderValue={(v) => v || (modelsLoading ? 'Loading...' : modelList[0] || 'No models (start backend)')}
                    >
                      <MenuItem value="" disabled>{modelList.length ? 'Select model' : 'No models (start backend)'}</MenuItem>
                      {modelList.map((m) => (
                        <MenuItem key={m} value={m}>{m}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>

              {/* Read only */}
              <Button
                fullWidth
                variant="outlined"
                startIcon={<ImageIcon />}
                onClick={handleReadImages}
                disabled={selectedFiles.length === 0 || isProcessing}
                size="large"
                sx={{ mb: 1 }}
              >
                {isProcessing ? 'Reading...' : 'Read images (OCR only)'}
              </Button>
              {/* Read & Process with AI (one or more images) */}
              <Button
                fullWidth
                variant="contained"
                startIcon={isProcessing ? <CircularProgress size={20} color="inherit" /> : <AIIcon />}
                onClick={handleReadAndProcess}
                disabled={selectedFiles.length === 0 || isProcessing || !effectiveModel || !systemPrompt?.trim()}
                size="large"
              >
                Read & process with AI
              </Button>

              {/* Processing Progress */}
              {isProcessing && totalImages > 1 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Processing image {currentImageIndex} of {totalImages}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(currentImageIndex / totalImages) * 100}
                    sx={{ mt: 1 }}
                  />
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
                  {processResult ? 'Read & AI result' : 'Extracted text'}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexShrink: 0 }}>
                  {(processResult || results.length > 0) && getMarkdownForSave() && (
                    <Button size="small" variant="outlined" startIcon={<SaveIcon />} onClick={handleSaveToFile}>
                      Save to file
                    </Button>
                  )}
                  {(processResult || results.length > 0) && getContentForRAG() && (
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

              {!isProcessing && !processResult && results.length === 0 && (
                <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary', flex: 1 }}>
                  <ImageIcon sx={{ fontSize: 64, mb: 2, opacity: 0.3 }} />
                  <Typography>No results yet. Use &quot;Read images&quot; or &quot;Read & process with AI&quot; (one image).</Typography>
                </Box>
              )}

              {!isProcessing && processResult && (
                <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <Tabs value={resultTab} onChange={(_, v) => setResultTab(v)} sx={{ mb: 2, flexShrink: 0 }}>
                    <Tab label="AI result" />
                    <Tab label="Extracted text" />
                  </Tabs>
                  {resultTab === 0 && (
                    <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default', flex: 1, minHeight: 0, overflow: 'auto' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                        {processResult.provider} / {processResult.model}
                      </Typography>
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
                        <ReactMarkdown>{processResult.ai_result || 'No AI result'}</ReactMarkdown>
                      </Box>
                    </Paper>
                  )}
                  {resultTab === 1 && (
                    <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default', flex: 1, minHeight: 0, overflow: 'auto' }}>
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                        {processResult.extracted_text || 'No text extracted'}
                      </Typography>
                    </Paper>
                  )}
                  {processResult.timestamp && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', flexShrink: 0 }}>
                      Processed: {new Date(processResult.timestamp).toLocaleString()}
                    </Typography>
                  )}
                </Box>
              )}

              {!isProcessing && !processResult && results.length > 0 && (
                <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto', pr: 0.5 }}>
                  {results.map((result, index) => (
                    <Paper
                      key={index}
                      sx={{
                        p: 2,
                        mb: 2,
                        border: '1px solid',
                        borderColor: result.success ? 'success.main' : 'error.main',
                        borderRadius: 1,
                      }}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                          {results.length > 1 ? `Image ${result.image_index || index + 1}` : 'Result'}
                        </Typography>
                        <Chip
                          label={result.success ? 'Success' : 'Error'}
                          color={result.success ? 'success' : 'error'}
                          size="small"
                        />
                      </Box>

                      {result.success ? (
                        <>
                          {result.image_info && (
                            <Box sx={{ mb: 1 }}>
                              <Typography variant="caption" color="text.secondary">
                                Size: {result.image_info.width} × {result.image_info.height} | Format:{' '}
                                {result.image_info.format}
                              </Typography>
                            </Box>
                          )}
                          <Divider sx={{ my: 1 }} />
                          <Typography
                            variant="body2"
                            sx={{
                              whiteSpace: 'pre-wrap',
                              fontFamily: 'monospace',
                              bgcolor: 'background.default',
                              p: 1,
                              borderRadius: 1,
                              maxHeight: 'min(360px, 40vh)',
                              overflow: 'auto',
                            }}
                          >
                            {result.text || 'No text extracted'}
                          </Typography>
                          {result.timestamp && (
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                              Processed: {new Date(result.timestamp).toLocaleString()}
                            </Typography>
                          )}
                        </>
                      ) : (
                        <Alert severity="error" sx={{ mt: 1 }}>
                          {result.error || 'Failed to read image'}
                        </Alert>
                      )}
                    </Paper>
                  ))}
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

export default ImageReader;
