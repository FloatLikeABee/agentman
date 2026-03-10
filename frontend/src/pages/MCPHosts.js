import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  IconButton,
  Alert,
  Chip,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Lan as NetworkIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';

const MCPHosts = () => {
  const queryClient = useQueryClient();
  const { data: hosts = [], isLoading, error } = useQuery('mcp-hosts', api.getMCPHosts);

  const [openDialog, setOpenDialog] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState({
    name: '',
    description: '',
    transport: 'tcp',
    // stdio
    command: '',
    args: '',
    env: '',
    working_dir: '',
    // tcp
    host: '127.0.0.1',
    port: 8196,
    // websocket
    url: '',
    is_active: true,
  });

  const resetForm = () => {
    setForm({
      name: '',
      description: '',
      transport: 'tcp',
      command: '',
      args: '',
      env: '',
      working_dir: '',
      host: '127.0.0.1',
      port: 8196,
      url: '',
      is_active: true,
    });
  };

  const createMutation = useMutation(api.createMCPHost, {
    onSuccess: () => {
      queryClient.invalidateQueries('mcp-hosts');
      setOpenDialog(false);
      resetForm();
    },
  });

  const updateMutation = useMutation(
    ({ hostId, payload }) => api.updateMCPHost(hostId, payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('mcp-hosts');
        setOpenDialog(false);
        setEditingId(null);
        resetForm();
      },
    }
  );

  const deleteMutation = useMutation(api.deleteMCPHost, {
    onSuccess: () => {
      queryClient.invalidateQueries('mcp-hosts');
    },
  });

  const openCreate = () => {
    setEditingId(null);
    resetForm();
    setOpenDialog(true);
  };

  const openEdit = (host) => {
    setEditingId(host.id);
    const cfg = host.config || {};
    setForm({
      name: host.name || cfg.name || '',
      description: host.description || cfg.description || '',
      transport: cfg.transport || 'tcp',
      command: cfg.command || '',
      args: (cfg.args && Array.isArray(cfg.args) ? cfg.args.join(' ') : ''),
      env:
        cfg.env && typeof cfg.env === 'object'
          ? Object.entries(cfg.env)
              .map(([k, v]) => `${k}=${v}`)
              .join('\n')
          : '',
      working_dir: cfg.working_dir || '',
      host: cfg.host || '127.0.0.1',
      port: cfg.port || 8196,
      url: cfg.url || '',
      is_active: cfg.is_active !== undefined ? cfg.is_active : true,
    });
    setOpenDialog(true);
  };

  const parseArgs = (value) =>
    value
      .split(/\s+/)
      .map((s) => s.trim())
      .filter(Boolean);

  const parseEnv = (value) => {
    const out = {};
    value
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .forEach((line) => {
        const idx = line.indexOf('=');
        if (idx > 0) {
          const key = line.substring(0, idx).trim();
          const val = line.substring(idx + 1).trim();
          if (key) out[key] = val;
        }
      });
    return out;
  };

  const buildPayload = () => {
    const base = {
      name: form.name,
      description: form.description || null,
      transport: form.transport,
      is_active: form.is_active,
      metadata: {},
    };
    const payload = { ...base };

    if (form.transport === 'stdio') {
      payload.command = form.command || null;
      payload.args = parseArgs(form.args || '');
      payload.env = parseEnv(form.env || '');
      payload.working_dir = form.working_dir || null;
      payload.host = null;
      payload.port = null;
      payload.url = null;
    } else if (form.transport === 'tcp') {
      payload.command = null;
      payload.args = [];
      payload.env = {};
      payload.working_dir = null;
      payload.host = form.host || '127.0.0.1';
      payload.port = form.port ? parseInt(form.port, 10) : 8196;
      payload.url = null;
    } else if (form.transport === 'websocket') {
      payload.command = null;
      payload.args = [];
      payload.env = {};
      payload.working_dir = null;
      payload.host = null;
      payload.port = null;
      payload.url = form.url || '';
    }
    return payload;
  };

  const handleSave = () => {
    const payload = buildPayload();
    if (editingId) {
      updateMutation.mutate({ hostId: editingId, payload });
    } else {
      createMutation.mutate(payload);
    }
  };

  const handleDelete = (hostId) => {
    deleteMutation.mutate(hostId);
  };

  const isFormValid = () => {
    if (!form.name.trim()) return false;
    if (form.transport === 'stdio') {
      return !!form.command.trim();
    }
    if (form.transport === 'tcp') {
      return !!form.host.trim() && !!String(form.port).trim();
    }
    if (form.transport === 'websocket') {
      return !!form.url.trim();
    }
    return true;
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4">MCP Hosts</Typography>
        <Button variant="contained" startIcon={<AddIcon />} onClick={openCreate}>
          New MCP Host
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load MCP hosts
        </Alert>
      )}

      <Grid container spacing={2}>
        {hosts.map((host) => {
          const cfg = host.config || {};
          const transport = cfg.transport || 'tcp';
          const active = cfg.is_active !== undefined ? cfg.is_active : true;
          let connectionLabel = '';
          if (transport === 'tcp') {
            connectionLabel = `${cfg.host || '127.0.0.1'}:${cfg.port || 8196}`;
          } else if (transport === 'websocket') {
            connectionLabel = cfg.url || '';
          } else if (transport === 'stdio') {
            connectionLabel = cfg.command || '';
          }
          return (
            <Grid item xs={12} md={6} key={host.id}>
              <Card sx={{ boxShadow: 2 }}>
                <CardContent sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Box sx={{ flex: 1, pr: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <NetworkIcon fontSize="small" />
                      <Typography variant="h6">{host.name}</Typography>
                      {!active && <Chip size="small" label="Inactive" />}
                    </Box>
                    {host.description && (
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                        {host.description}
                      </Typography>
                    )}
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 0.5 }}>
                      <Chip
                        size="small"
                        label={transport.toUpperCase()}
                        color={transport === 'stdio' ? 'secondary' : transport === 'websocket' ? 'info' : 'primary'}
                        variant="outlined"
                      />
                      {connectionLabel && (
                        <Chip size="small" label={connectionLabel} variant="outlined" />
                      )}
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    <IconButton
                      size="small"
                      color="primary"
                      onClick={() => openEdit(host)}
                    >
                      <EditIcon fontSize="small" />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleDelete(host.id)}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
        {!isLoading && hosts.length === 0 && (
          <Grid item xs={12}>
            <Typography variant="body2" color="text.secondary">
              No MCP host configurations yet. Create one to get started.
            </Typography>
          </Grid>
        )}
      </Grid>

      <Dialog
        open={openDialog}
        onClose={() => {
          setOpenDialog(false);
          setEditingId(null);
          resetForm();
        }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>{editingId ? 'Edit MCP Host' : 'Create MCP Host'}</DialogTitle>
        <DialogContent>
          {(createMutation.isError || updateMutation.isError) && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to save MCP host. Please check your inputs and try again.
            </Alert>
          )}
          <Box sx={{ mt: 1 }}>
            <TextField
              fullWidth
              label="Name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              sx={{ mb: 2 }}
              required
            />
            <TextField
              fullWidth
              label="Description"
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
              multiline
              rows={2}
              sx={{ mb: 2 }}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Transport</InputLabel>
              <Select
                value={form.transport}
                label="Transport"
                onChange={(e) => setForm({ ...form, transport: e.target.value })}
              >
                <MenuItem value="tcp">TCP (host/port)</MenuItem>
                <MenuItem value="websocket">WebSocket URL</MenuItem>
                <MenuItem value="stdio">Stdio (spawn process)</MenuItem>
              </Select>
            </FormControl>

            {form.transport === 'tcp' && (
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={12} sm={8}>
                  <TextField
                    fullWidth
                    label="Host"
                    value={form.host}
                    onChange={(e) => setForm({ ...form, host: e.target.value })}
                    required
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    label="Port"
                    type="number"
                    value={form.port}
                    onChange={(e) => setForm({ ...form, port: e.target.value })}
                    required
                  />
                </Grid>
              </Grid>
            )}

            {form.transport === 'websocket' && (
              <TextField
                fullWidth
                label="WebSocket URL"
                value={form.url}
                onChange={(e) => setForm({ ...form, url: e.target.value })}
                sx={{ mb: 2 }}
                required
                placeholder="wss://example.com/mcp"
              />
            )}

            {form.transport === 'stdio' && (
              <>
                <TextField
                  fullWidth
                  label="Command"
                  value={form.command}
                  onChange={(e) => setForm({ ...form, command: e.target.value })}
                  sx={{ mb: 2 }}
                  required
                  placeholder="python -m my_mcp_server"
                />
                <TextField
                  fullWidth
                  label="Arguments (space separated)"
                  value={form.args}
                  onChange={(e) => setForm({ ...form, args: e.target.value })}
                  sx={{ mb: 2 }}
                  placeholder="--port 8196 --config config.yml"
                />
                <TextField
                  fullWidth
                  label="Environment Variables (one KEY=VALUE per line)"
                  value={form.env}
                  onChange={(e) => setForm({ ...form, env: e.target.value })}
                  sx={{ mb: 2 }}
                  multiline
                  rows={4}
                  placeholder={'MCP_API_KEY=...\nOTHER_VAR=value'}
                />
                <TextField
                  fullWidth
                  label="Working Directory (optional)"
                  value={form.working_dir}
                  onChange={(e) => setForm({ ...form, working_dir: e.target.value })}
                  sx={{ mb: 2 }}
                  placeholder="C:\\path\\to\\project"
                />
              </>
            )}

            <FormControlLabel
              control={
                <Switch
                  checked={form.is_active}
                  onChange={(e) => setForm({ ...form, is_active: e.target.checked })}
                />
              }
              label="Active"
              sx={{ mb: 2 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              setOpenDialog(false);
              setEditingId(null);
              resetForm();
            }}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSave}
            disabled={createMutation.isLoading || updateMutation.isLoading || !isFormValid()}
          >
            {createMutation.isLoading || updateMutation.isLoading
              ? 'Saving...'
              : editingId
              ? 'Update'
              : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MCPHosts;

