import React, { useState } from 'react';
import {
  Box,
  Drawer,
  IconButton,
  TextField,
  Typography,
  Button,
  CircularProgress,
  Divider,
  Tooltip,
  Fab,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import { HelpOutline as HelpIcon, Close as CloseIcon } from '@mui/icons-material';
import { useMutation } from 'react-query';
import ReactMarkdown from 'react-markdown';
import api from '../services/api';

const TheHelp = () => {
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [history, setHistory] = useState([]); // {question, answer, sources}
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const askMutation = useMutation(api.askHelp, {
    onSuccess: (data) => {
      setHistory((prev) => [
        ...prev,
        {
          question,
          answer: data?.answer || '',
          sources: data?.sources || [],
          used_rag: data?.used_rag,
          error: data?.error || null,
        },
      ]);
      setQuestion('');
    },
  });

  const handleAsk = () => {
    if (!question.trim()) return;
    askMutation.mutate({ question: question.trim() });
  };

  return (
    <>
      {/* Floating button */}
      <Box
        sx={{
          position: 'fixed',
          right: 16,
          bottom: 24,
          zIndex: 1400,
        }}
      >
        <Tooltip title="The Help – Ask how the system works" arrow>
          {isMobile ? (
            <IconButton
              color="primary"
              onClick={() => setOpen(true)}
              sx={{
                bgcolor: 'background.paper',
                boxShadow: 6,
                border: '1px solid',
                borderColor: 'primary.main',
                '&:hover': {
                  boxShadow: 10,
                },
              }}
            >
              <HelpIcon />
            </IconButton>
          ) : (
            <Fab
              variant="extended"
              color="primary"
              onClick={() => setOpen(true)}
              sx={{
                boxShadow: 10,
                border: '1px solid',
                borderColor: 'primary.main',
                fontWeight: 700,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
              }}
            >
              <HelpIcon sx={{ mr: 1 }} />
              Help
            </Fab>
          )}
        </Tooltip>
      </Box>

      {/* Side drawer */}
      <Drawer
        anchor="right"
        open={open}
        onClose={() => setOpen(false)}
        PaperProps={{
          sx: {
            width: { xs: '100%', sm: 420, md: 480 },
            maxWidth: '90vw',
            display: 'flex',
            flexDirection: 'column',
          },
        }}
      >
        <Box
          sx={{
            p: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <HelpIcon color="primary" />
            <Typography variant="h6">The Help</Typography>
          </Box>
          <IconButton size="small" onClick={() => setOpen(false)}>
            <CloseIcon />
          </IconButton>
        </Box>

        {/* Query input */}
        <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Ask anything about how this system works: modules, flows, configuration, or APIs.
          </Typography>
          <TextField
            fullWidth
            multiline
            minRows={2}
            maxRows={4}
            size="small"
            placeholder="e.g. How do I set up a new RAG collection and attach it to an agent?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            sx={{ mb: 1.5 }}
          />
          <Button
            variant="contained"
            fullWidth
            onClick={handleAsk}
            disabled={askMutation.isLoading || !question.trim()}
            startIcon={askMutation.isLoading ? <CircularProgress size={18} color="inherit" /> : null}
          >
            {askMutation.isLoading ? 'Thinking...' : 'Ask The Help'}
          </Button>
        </Box>

        {/* History / answers */}
        <Box
          sx={{
            flex: 1,
            overflowY: 'auto',
            p: 2,
          }}
        >
          {history.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              No questions yet. Ask The Help anything about how to use Ground Control.
            </Typography>
          )}

          {history.map((item, idx) => (
            <Box key={idx} sx={{ mb: 3 }}>
              <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                You
              </Typography>
              <Box
                sx={{
                  p: 1.5,
                  borderRadius: 1,
                  bgcolor: 'background.default',
                  mb: 1,
                  border: '1px solid',
                  borderColor: 'divider',
                }}
              >
                <Typography variant="body2">{item.question}</Typography>
              </Box>

              <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                The Help
              </Typography>
              <Box
                sx={{
                  p: 1.5,
                  borderRadius: 1,
                  bgcolor: 'background.paper',
                  border: '1px solid',
                  borderColor: 'primary.main',
                  borderOpacity: 0.3,
                }}
              >
                {item.error && (
                  <Typography variant="body2" color="error" sx={{ mb: 1 }}>
                    {item.error === 'system_help_collection_missing'
                      ? 'The help documentation collection is not initialized yet. Add system docs to the RAG collection named "system_help".'
                      : `Error: ${item.error}`}
                  </Typography>
                )}
                <ReactMarkdown>{item.answer || 'No answer.'}</ReactMarkdown>

                {item.sources && item.sources.length > 0 && (
                  <>
                    <Divider sx={{ my: 1.5 }} />
                    <Typography variant="caption" color="text.secondary">
                      Sources used ({item.used_rag ? 'RAG enabled' : 'no RAG context'}):
                    </Typography>
                    {item.sources.slice(0, 5).map((src, i) => (
                      <Typography key={i} variant="caption" display="block">
                        • {src.collection} – {src.document_id || '(doc)'}
                      </Typography>
                    ))}
                  </>
                )}
              </Box>
            </Box>
          ))}
        </Box>
      </Drawer>
    </>
  );
};

export default TheHelp;

