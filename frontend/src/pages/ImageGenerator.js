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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Tooltip,
  Paper,
  ImageList,
  ImageListItem,
  ImageListItemBar,
  Chip,
  Divider,
} from '@mui/material';
import {
  Image as ImageIcon,
  AutoFixHigh as PolishIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Close as CloseIcon,
  Fullscreen as FullscreenIcon,
  ContentCopy as CopyIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';

const ImageGenerator = () => {
  const queryClient = useQueryClient();
  const [prompt, setPrompt] = useState('');
  const [polishedPrompt, setPolishedPrompt] = useState('');
  const [usePolished, setUsePolished] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState('qwen');
  const [selectedModel, setSelectedModel] = useState('');
  const [generatedImageUrl, setGeneratedImageUrl] = useState(null);
  const [imageDialogOpen, setImageDialogOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [error, setError] = useState(null);
  const [isPolishing, setIsPolishing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

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

  // Fetch saved images
  const { data: savedImages = [], isLoading: imagesLoading, refetch: refetchImages } = useQuery(
    'generatedImages',
    api.getGeneratedImages,
    {
      refetchOnWindowFocus: false,
    }
  );

  // Set default model when provider changes
  useEffect(() => {
    if (models[selectedProvider] && models[selectedProvider].length > 0) {
      setSelectedModel(models[selectedProvider][0]);
    }
  }, [selectedProvider, models]);

  // Polish prompt mutation
  const polishPromptMutation = useMutation(
    (data) => api.polishImagePrompt(data),
    {
      onSuccess: (data) => {
        setPolishedPrompt(data.polished_prompt);
        setUsePolished(true);
        setIsPolishing(false);
      },
      onError: (err) => {
        setError(err.response?.data?.detail || 'Failed to polish prompt');
        setIsPolishing(false);
      },
    }
  );

  // Generate image mutation
  const generateImageMutation = useMutation(
    (data) => api.generateImage(data),
    {
      onSuccess: (data) => {
        setGeneratedImageUrl(data.image_url);
        setImageDialogOpen(true);
        setIsGenerating(false);
        refetchImages();
      },
      onError: (err) => {
        setError(err.response?.data?.detail || 'Failed to generate image');
        setIsGenerating(false);
      },
    }
  );

  // Delete image mutation
  const deleteImageMutation = useMutation(
    (filename) => api.deleteGeneratedImage(filename),
    {
      onSuccess: () => {
        refetchImages();
      },
      onError: (err) => {
        setError(err.response?.data?.detail || 'Failed to delete image');
      },
    }
  );

  const handlePolishPrompt = () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt first');
      return;
    }
    setError(null);
    setIsPolishing(true);
    polishPromptMutation.mutate({
      prompt: prompt,
      provider: selectedProvider,
      model: selectedModel,
    });
  };

  const handleGenerateImage = () => {
    const finalPrompt = usePolished && polishedPrompt ? polishedPrompt : prompt;
    if (!finalPrompt.trim()) {
      setError('Please enter a prompt');
      return;
    }
    setError(null);
    setIsGenerating(true);
    generateImageMutation.mutate({
      prompt: finalPrompt,
      save: true,
    });
  };

  const handleImageClick = (image) => {
    setSelectedImage(image);
    setImageDialogOpen(true);
  };

  const handleDeleteImage = (filename, e) => {
    e.stopPropagation();
    if (window.confirm('Delete this image?')) {
      deleteImageMutation.mutate(filename);
    }
  };

  const handleDownloadImage = async (imageUrl, filename) => {
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename || 'generated-image.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to download image');
    }
  };

  const handleCopyUrl = (url) => {
    navigator.clipboard.writeText(url);
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
        <ImageIcon /> Image Generator
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Generate AI images using Pollinations API. Optionally polish your prompt with AI first.
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
            {/* Prompt Input */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Image Prompt"
                placeholder="Describe the image you want to generate... e.g., 'a futuristic city at sunset with flying cars'"
                value={prompt}
                onChange={(e) => {
                  setPrompt(e.target.value);
                  setUsePolished(false);
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&:hover fieldset': {
                      borderColor: '#9d4edd',
                    },
                  },
                }}
              />
            </Grid>

            {/* AI Polish Section */}
            <Grid item xs={12}>
              <Paper 
                variant="outlined" 
                sx={{ 
                  p: 2, 
                  bgcolor: 'rgba(157, 78, 221, 0.05)',
                  borderColor: 'rgba(157, 78, 221, 0.2)',
                }}
              >
                <Typography variant="subtitle2" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <PolishIcon fontSize="small" color="primary" />
                  AI Prompt Enhancement (Optional)
                </Typography>
                
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} sm={4}>
                    <FormControl fullWidth size="small">
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
                  
                  <Grid item xs={12} sm={4}>
                    <FormControl fullWidth size="small">
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
                  
                  <Grid item xs={12} sm={4}>
                    <Button
                      fullWidth
                      variant="outlined"
                      color="secondary"
                      startIcon={isPolishing ? <CircularProgress size={20} /> : <PolishIcon />}
                      onClick={handlePolishPrompt}
                      disabled={isPolishing || !prompt.trim()}
                    >
                      {isPolishing ? 'Polishing...' : 'Polish Prompt'}
                    </Button>
                  </Grid>
                </Grid>

                {/* Polished Prompt Display */}
                {polishedPrompt && (
                  <Box sx={{ mt: 2 }}>
                    <Divider sx={{ mb: 2 }} />
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <Typography variant="subtitle2" color="primary">
                        Polished Prompt:
                      </Typography>
                      <Chip 
                        size="small" 
                        label={usePolished ? "Using this" : "Click to use"} 
                        color={usePolished ? "primary" : "default"}
                        onClick={() => setUsePolished(!usePolished)}
                        sx={{ cursor: 'pointer' }}
                      />
                    </Box>
                    <Paper 
                      variant="outlined" 
                      sx={{ 
                        p: 2, 
                        bgcolor: usePolished ? 'rgba(157, 78, 221, 0.1)' : 'transparent',
                        borderColor: usePolished ? '#9d4edd' : 'rgba(255,255,255,0.1)',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        '&:hover': {
                          borderColor: '#9d4edd',
                        },
                      }}
                      onClick={() => setUsePolished(!usePolished)}
                    >
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                        {polishedPrompt}
                      </Typography>
                    </Paper>
                  </Box>
                )}
              </Paper>
            </Grid>

            {/* Generate Button */}
            <Grid item xs={12}>
              <Button
                fullWidth
                variant="contained"
                size="large"
                startIcon={isGenerating ? <CircularProgress size={24} color="inherit" /> : <ImageIcon />}
                onClick={handleGenerateImage}
                disabled={isGenerating || (!prompt.trim() && !polishedPrompt)}
                sx={{
                  py: 1.5,
                  fontSize: '1.1rem',
                  background: 'linear-gradient(135deg, #9d4edd 0%, #7b2cbf 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #c77dff 0%, #9d4edd 100%)',
                  },
                }}
              >
                {isGenerating ? 'Generating...' : `Generate Image${usePolished ? ' (with polished prompt)' : ''}`}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Generated Images Gallery */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <ImageIcon /> Generated Images
            </Typography>
            <Tooltip title="Refresh">
              <IconButton onClick={() => refetchImages()} size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>

          {imagesLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : savedImages.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
              No images generated yet. Create your first image above!
            </Typography>
          ) : (
            <ImageList cols={3} gap={16}>
              {savedImages.map((image) => (
                <ImageListItem 
                  key={image.filename}
                  sx={{ 
                    cursor: 'pointer',
                    borderRadius: 2,
                    overflow: 'hidden',
                    border: '1px solid rgba(157, 78, 221, 0.2)',
                    transition: 'all 0.3s',
                    '&:hover': {
                      transform: 'scale(1.02)',
                      borderColor: '#9d4edd',
                      boxShadow: '0 4px 20px rgba(157, 78, 221, 0.3)',
                    },
                  }}
                  onClick={() => handleImageClick(image)}
                >
                  <img
                    src={image.url}
                    alt={image.prompt || 'Generated image'}
                    loading="lazy"
                    style={{ 
                      height: 200, 
                      objectFit: 'cover',
                    }}
                  />
                  <ImageListItemBar
                    title={image.prompt ? (image.prompt.length > 40 ? image.prompt.substring(0, 40) + '...' : image.prompt) : 'Image'}
                    subtitle={image.created_at}
                    actionIcon={
                      <Box sx={{ display: 'flex', gap: 0.5, pr: 1 }}>
                        <Tooltip title="Download">
                          <IconButton
                            size="small"
                            sx={{ color: 'white' }}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDownloadImage(image.url, image.filename);
                            }}
                          >
                            <DownloadIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton
                            size="small"
                            sx={{ color: 'white' }}
                            onClick={(e) => handleDeleteImage(image.filename, e)}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    }
                    sx={{
                      background: 'linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0) 100%)',
                    }}
                  />
                </ImageListItem>
              ))}
            </ImageList>
          )}
        </CardContent>
      </Card>

      {/* Image Preview Dialog */}
      <Dialog 
        open={imageDialogOpen} 
        onClose={() => {
          setImageDialogOpen(false);
          setSelectedImage(null);
          setGeneratedImageUrl(null);
        }}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">
            {selectedImage ? 'Image Details' : 'Generated Image'}
          </Typography>
          <IconButton onClick={() => {
            setImageDialogOpen(false);
            setSelectedImage(null);
            setGeneratedImageUrl(null);
          }}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ textAlign: 'center' }}>
            <img
              src={selectedImage?.url || generatedImageUrl}
              alt="Generated"
              style={{ 
                maxWidth: '100%', 
                maxHeight: '60vh',
                borderRadius: 8,
                boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
              }}
            />
            {(selectedImage?.prompt || (usePolished ? polishedPrompt : prompt)) && (
              <Paper variant="outlined" sx={{ mt: 2, p: 2, textAlign: 'left' }}>
                <Typography variant="subtitle2" color="primary" gutterBottom>
                  Prompt:
                </Typography>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {selectedImage?.prompt || (usePolished ? polishedPrompt : prompt)}
                </Typography>
              </Paper>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button
            startIcon={<CopyIcon />}
            onClick={() => handleCopyUrl(selectedImage?.url || generatedImageUrl)}
          >
            Copy URL
          </Button>
          <Button
            startIcon={<DownloadIcon />}
            onClick={() => handleDownloadImage(
              selectedImage?.url || generatedImageUrl, 
              selectedImage?.filename || 'generated-image.png'
            )}
          >
            Download
          </Button>
          <Button onClick={() => {
            setImageDialogOpen(false);
            setSelectedImage(null);
            setGeneratedImageUrl(null);
          }}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ImageGenerator;
