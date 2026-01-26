import React, { useState } from 'react';
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
} from '@mui/material';
import {
  Image as ImageIcon,
  Upload as UploadIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import api from '../services/api';

const ImageReader = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [results, setResults] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [totalImages, setTotalImages] = useState(0);

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
    setError(null);
    setCurrentImageIndex(0);
    setTotalImages(0);
  };

  const handleReadImages = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResults([]);
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

      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid item xs={12} md={6}>
          <Card>
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

              {/* Custom Prompt */}
              <TextField
                fullWidth
                label="Custom Prompt (Optional)"
                placeholder="Please output only the text content from the image without any additional descriptions or formatting."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                multiline
                rows={3}
                sx={{ mb: 2 }}
                helperText="Leave empty to use default OCR extraction prompt"
              />

              {/* Read Button */}
              <Button
                fullWidth
                variant="contained"
                startIcon={<ImageIcon />}
                onClick={handleReadImages}
                disabled={selectedFiles.length === 0 || isProcessing}
                size="large"
              >
                {isProcessing ? 'Reading Images...' : 'Read Images'}
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
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Extracted Text
              </Typography>

              {error && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
                  {error}
                </Alert>
              )}

              {isProcessing && (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              )}

              {!isProcessing && results.length === 0 && (
                <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                  <ImageIcon sx={{ fontSize: 64, mb: 2, opacity: 0.3 }} />
                  <Typography>No results yet. Upload and read images to see extracted text.</Typography>
                </Box>
              )}

              {!isProcessing && results.length > 0 && (
                <Box>
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
                              maxHeight: '400px',
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
    </Box>
  );
};

export default ImageReader;
