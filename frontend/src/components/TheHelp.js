import React, { useState } from 'react';
import {
  Box,
  Drawer,
  Card,
  CardActionArea,
  CardContent,
  IconButton,
  TextField,
  Typography,
  Button,
  CircularProgress,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  SmartToy as RobotIcon,
  Close as CloseIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';
import { useMutation } from 'react-query';
import ReactMarkdown from 'react-markdown';
import api from '../services/api';

const TheHelp = () => {
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [history, setHistory] = useState([]); // {question, answer, sources}
  const [expandedIdx, setExpandedIdx] = useState(null);

  const shorten = (text, maxLen = 140) => {
    if (!text) return '';
    const oneLine = text.replace(/\s+/g, ' ').trim();
    return oneLine.length > maxLen ? `${oneLine.slice(0, maxLen - 1)}…` : oneLine;
  };

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
      setExpandedIdx(history.length);
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
          <IconButton
            aria-label="Open The Help"
            color="primary"
            onClick={() => setOpen(true)}
            sx={{
              width: 52,
              height: 52,
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: 'primary.main',
              boxShadow: '0 0 18px rgba(157, 78, 221, 0.35), 0 10px 24px rgba(0,0,0,0.5)',
              '&:hover': {
                boxShadow: '0 0 26px rgba(157, 78, 221, 0.55), 0 14px 30px rgba(0,0,0,0.6)',
                transform: 'translateY(-1px)',
              },
              transition: 'transform 120ms ease, box-shadow 200ms ease',
            }}
          >
            <RobotIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Side drawer */}
      <Drawer
        anchor="right"
        open={open}
        onClose={() => setOpen(false)}
        PaperProps={{
          sx: {
            width: { xs: '100%', md: '50vw' },
            minWidth: { md: 700 },
            maxWidth: '100vw',
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
            <RobotIcon color="primary" />
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
            <Card key={idx} sx={{ mb: 1.5, border: '1px solid', borderColor: 'divider' }}>
              <CardActionArea onClick={() => setExpandedIdx(expandedIdx === idx ? null : idx)}>
                <CardContent sx={{ pb: expandedIdx === idx ? 1.5 : 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 1 }}>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography variant="subtitle2" color="primary.main" sx={{ mb: 0.5 }}>
                        {shorten(item.question, 120)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {shorten(item.answer || 'No answer.', 180)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.75, display: 'block' }}>
                        {item.used_rag ? 'RAG used' : 'No RAG'} • {item.sources?.length || 0} sources
                      </Typography>
                    </Box>
                    <Box sx={{ pt: 0.5 }}>
                      {expandedIdx === idx ? (
                        <ExpandLessIcon fontSize="small" />
                      ) : (
                        <ExpandMoreIcon fontSize="small" />
                      )}
                    </Box>
                  </Box>
                </CardContent>
              </CardActionArea>

              {expandedIdx === idx && (
                <Box sx={{ px: 2, pb: 2 }}>
                  <Divider sx={{ mb: 1.5 }} />
                  <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                    You
                  </Typography>
                  <Box
                    sx={{
                      p: 1.5,
                      borderRadius: 1,
                      bgcolor: 'background.default',
                      mb: 1.5,
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
              )}
            </Card>
          ))}
        </Box>
      </Drawer>
    </>
  );
};

export default TheHelp;

