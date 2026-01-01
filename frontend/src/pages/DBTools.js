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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Tooltip,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Visibility as PreviewIcon,
  Refresh as RefreshIcon,
  Storage as DatabaseIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';

const DBTools = () => {
  const queryClient = useQueryClient();
  const { data: tools = [], isLoading, error } = useQuery('db-tools', api.getDBTools);

  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [editingToolId, setEditingToolId] = useState(null);
  const [selectedTool, setSelectedTool] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [createForm, setCreateForm] = useState({
    name: '',
    description: '',
    db_type: 'mysql',
    host: '',
    port: 3306,
    database: '',
    username: '',
    password: '',
    sql_statement: '',
    is_active: true,
    cache_ttl_hours: 1.0,
    allow_dynamic_sql: false,
    preset_sql_statement: '',
    additional_params: {},
  });
  const [sqlInput, setSqlInput] = useState('');
  const [executeLoading, setExecuteLoading] = useState(false);

  const createMutation = useMutation(api.createDBTool, {
    onSuccess: () => {
      queryClient.invalidateQueries('db-tools');
      setOpenCreateDialog(false);
      resetForm();
    },
  });

  const updateMutation = useMutation(
    ({ toolId, payload }) => api.updateDBTool(toolId, payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('db-tools');
        setOpenEditDialog(false);
        setEditingToolId(null);
        resetForm();
        if (selectedTool && selectedTool.id === editingToolId) {
          setSelectedTool(null);
          setPreviewData(null);
        }
      },
    }
  );

  const deleteMutation = useMutation(api.deleteDBTool, {
    onSuccess: () => {
      queryClient.invalidateQueries('db-tools');
      if (selectedTool) {
        setSelectedTool(null);
        setPreviewData(null);
      }
    },
  });

  const resetForm = () => {
    setCreateForm({
      name: '',
      description: '',
      db_type: 'mysql',
      host: '',
      port: 3306,
      database: '',
      username: '',
      password: '',
      sql_statement: '',
      is_active: true,
      cache_ttl_hours: 1.0,
      allow_dynamic_sql: false,
      preset_sql_statement: '',
      additional_params: {},
    });
    setSqlInput('');
  };

  const getDefaultPort = (dbType) => {
    switch (dbType) {
      case 'mysql':
        return 3306;
      case 'sqlserver':
        return 1433;
      case 'sqlite':
        return 0; // SQLite doesn't use port
      case 'mongodb':
        return 27017;
      default:
        return 3306;
    }
  };

  const handleCreate = () => {
    // For SQLite, use database field as file path, or host if database is empty
    const dbPath = createForm.db_type === 'sqlite' 
      ? (createForm.database || createForm.host)
      : createForm.database;
    
    const payload = {
      name: createForm.name,
      description: createForm.description || null,
      db_type: createForm.db_type,
      connection_config: {
        host: createForm.db_type === 'sqlite' ? dbPath : createForm.host,
        port: createForm.db_type === 'sqlite' ? 0 : createForm.port,
        database: dbPath,
        username: createForm.db_type === 'sqlite' ? '' : createForm.username,
        password: createForm.db_type === 'sqlite' ? '' : createForm.password,
        additional_params: createForm.additional_params || {},
      },
      sql_statement: createForm.sql_statement,
      is_active: createForm.is_active,
      cache_ttl_hours: createForm.cache_ttl_hours,
      allow_dynamic_sql: createForm.allow_dynamic_sql,
      preset_sql_statement: createForm.allow_dynamic_sql ? (createForm.preset_sql_statement || '') : '',
      metadata: {},
    };
    createMutation.mutate(payload);
  };

  const handleEditTool = (tool) => {
    setEditingToolId(tool.id);
    setCreateForm({
      name: tool.name || '',
      description: tool.description || '',
      db_type: tool.db_type || 'mysql',
      host: tool.connection_config?.host || '',
      port: tool.connection_config?.port || getDefaultPort(tool.db_type),
      database: tool.connection_config?.database || '',
      username: tool.connection_config?.username || '',
      password: '', // Empty - user can leave blank to keep existing or enter new password
      sql_statement: tool.sql_statement || '',
      is_active: tool.is_active !== undefined ? tool.is_active : true,
      cache_ttl_hours: tool.cache_ttl_hours || 1.0,
      allow_dynamic_sql: tool.allow_dynamic_sql || false,
      preset_sql_statement: tool.preset_sql_statement || '',
      additional_params: tool.connection_config?.additional_params || {},
    });
    setOpenEditDialog(true);
  };

  const handleUpdate = () => {
    // For SQLite, use database field as file path, or host if database is empty
    const dbPath = createForm.db_type === 'sqlite' 
      ? (createForm.database || createForm.host)
      : createForm.database;
    
    const connectionConfig = {
      host: createForm.db_type === 'sqlite' ? dbPath : createForm.host,
      port: createForm.db_type === 'sqlite' ? 0 : createForm.port,
      database: dbPath,
      username: createForm.db_type === 'sqlite' ? '' : createForm.username,
      additional_params: createForm.additional_params || {},
    };
    
    // Only include password if provided (empty/null means keep existing)
    // For SQLite, password is not used
    if (createForm.db_type !== 'sqlite') {
      if (createForm.password && createForm.password.trim() !== '') {
        connectionConfig.password = createForm.password;
      } else {
        connectionConfig.password = null; // Backend will preserve existing
      }
    } else {
      connectionConfig.password = null;
    }
    
    const payload = {
      name: createForm.name,
      description: createForm.description || null,
      db_type: createForm.db_type,
      connection_config: connectionConfig,
      sql_statement: createForm.sql_statement,
      is_active: createForm.is_active,
      cache_ttl_hours: createForm.cache_ttl_hours,
      allow_dynamic_sql: createForm.allow_dynamic_sql,
      preset_sql_statement: createForm.allow_dynamic_sql ? (createForm.preset_sql_statement || '') : '',
      metadata: {},
    };
    
    updateMutation.mutate({ toolId: editingToolId, payload });
  };

  const handleDeleteTool = (toolId) => {
    if (window.confirm('Are you sure you want to delete this database tool? This action cannot be undone.')) {
      deleteMutation.mutate(toolId);
    }
  };

  const handlePreview = async (toolId, forceRefresh = false) => {
    setPreviewLoading(true);
    setPreviewError('');
    try {
      const result = await api.previewDBTool(toolId, forceRefresh);
      setPreviewData(result);
      setPreviewError('');
    } catch (err) {
      console.error('Preview error:', err);
      setPreviewError(err.response?.data?.detail || err.message || 'Failed to preview data');
      setPreviewData(null);
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleSelectTool = (tool) => {
    setSelectedTool(tool);
    setPreviewData(null);
    setPreviewError('');
    setSqlInput('');
    // Auto-load preview when tool is selected
    if (tool.is_active) {
      handlePreview(tool.id, false);
    }
  };

  const handleExecute = async () => {
    if (!selectedTool || !selectedTool.is_active) return;
    
    setExecuteLoading(true);
    setPreviewError('');
    try {
      const result = await api.executeDBTool(selectedTool.id, sqlInput || null);
      setPreviewData(result);
      setPreviewError('');
    } catch (err) {
      console.error('Execute error:', err);
      setPreviewError(err.response?.data?.detail || err.message || 'Failed to execute query');
      setPreviewData(null);
    } finally {
      setExecuteLoading(false);
    }
  };

  const getDBTypeColor = (dbType) => {
    switch (dbType) {
      case 'mysql':
        return 'success';
      case 'sqlserver':
        return 'primary';
      case 'sqlite':
        return 'info';
      case 'mongodb':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getDBTypeLabel = (dbType) => {
    switch (dbType) {
      case 'mysql':
        return 'MySQL';
      case 'sqlserver':
        return 'SQL Server';
      case 'sqlite':
        return 'SQLite';
      case 'mongodb':
        return 'MongoDB';
      default:
        return dbType;
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Database Tools</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            resetForm();
            setOpenCreateDialog(true);
          }}
        >
          New DB Tool
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load database tools
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={5}>
          <Grid container spacing={2}>
            {tools.map((tool) => (
              <Grid item xs={12} key={tool.id}>
                <Card
                  sx={{
                    boxShadow: 2,
                    cursor: 'pointer',
                    border:
                      selectedTool && selectedTool.id === tool.id
                        ? '2px solid #1976d2'
                        : '1px solid rgba(0,0,0,0.12)',
                  }}
                  onClick={() => handleSelectTool(tool)}
                >
                  <CardContent sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Box sx={{ flex: 1, pr: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <DatabaseIcon fontSize="small" />
                        <Typography variant="h6">{tool.name}</Typography>
                        {!tool.is_active && (
                          <Chip size="small" label="Inactive" color="default" />
                        )}
                      </Box>
                      {tool.description && (
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                          {tool.description}
                        </Typography>
                      )}
                      <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        <Chip
                          size="small"
                          label={getDBTypeLabel(tool.db_type)}
                          color={getDBTypeColor(tool.db_type)}
                          variant="outlined"
                        />
                        <Chip
                          size="small"
                          label={`Cache: ${tool.cache_ttl_hours}h`}
                          variant="outlined"
                        />
                        {tool.allow_dynamic_sql && (
                          <Chip
                            size="small"
                            label="Dynamic SQL"
                            color="secondary"
                            variant="outlined"
                          />
                        )}
                        <Chip
                          size="small"
                          label={`${tool.connection_config?.host || 'N/A'}:${tool.connection_config?.port || 'N/A'}`}
                          variant="outlined"
                        />
                      </Box>
                    </Box>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Tooltip title="Preview Data">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={(e) => {
                            e.stopPropagation();
                            handlePreview(tool.id, false);
                            handleSelectTool(tool);
                          }}
                        >
                          <PreviewIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEditTool(tool);
                        }}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        color="error"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteTool(tool.id);
                        }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
            {!isLoading && tools.length === 0 && (
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  No database tools yet. Create one to get started.
                </Typography>
              </Grid>
            )}
          </Grid>
        </Grid>

        {/* Preview Panel */}
        <Grid item xs={12} md={7}>
          <Card sx={{ boxShadow: 2, height: '100%' }}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Data Preview</Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  {selectedTool && selectedTool.is_active && (
                    <>
                      <Button
                        size="small"
                        startIcon={<RefreshIcon />}
                        onClick={() => handlePreview(selectedTool.id, true)}
                        disabled={previewLoading}
                      >
                        Refresh
                      </Button>
                    </>
                  )}
                </Box>
              </Box>
              {selectedTool ? (
                <>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'medium', mb: 1 }}>
                    {selectedTool.name}
                  </Typography>
                  {!selectedTool.is_active && (
                    <Alert severity="warning" sx={{ mb: 2 }}>
                      This database tool is inactive. Activate it to preview data.
                    </Alert>
                  )}
                  {selectedTool.is_active && (
                    <>
                      {selectedTool.allow_dynamic_sql && (
                        <Box sx={{ mb: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
                          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
                            Execute with Dynamic SQL
                          </Typography>
                          <TextField
                            fullWidth
                            label="SQL Input (WHERE condition or full SQL)"
                            value={sqlInput}
                            onChange={(e) => setSqlInput(e.target.value)}
                            multiline
                            rows={3}
                            sx={{ mb: 1 }}
                            placeholder={
                              selectedTool.preset_sql_statement
                                ? 'active = 1 AND created_at > \'2024-01-01\''
                                : 'SELECT * FROM orders WHERE status = \'pending\''
                            }
                            helperText={
                              selectedTool.preset_sql_statement
                                ? `Will be appended to: ${selectedTool.preset_sql_statement.substring(0, 50)}...`
                                : 'Enter full SQL statement or WHERE condition'
                            }
                          />
                          <Button
                            variant="contained"
                            onClick={handleExecute}
                            disabled={executeLoading || !sqlInput.trim()}
                            sx={{ mt: 1 }}
                          >
                            {executeLoading ? 'Executing...' : 'Execute Query'}
                          </Button>
                        </Box>
                      )}
                      {previewLoading || executeLoading ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                          <CircularProgress />
                        </Box>
                      ) : previewError ? (
                        <Alert severity="error" sx={{ mb: 2 }}>
                          {previewError}
                        </Alert>
                      ) : previewData ? (
                        <Box>
                          <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                            <Chip
                              size="small"
                              label={`Total Rows: ${previewData.total_rows || 0}`}
                              color="primary"
                            />
                            <Chip
                              size="small"
                              label={previewData.cached ? 'Cached' : 'Fresh'}
                              color={previewData.cached ? 'default' : 'success'}
                            />
                            {previewData.cache_expires_at && (
                              <Typography variant="caption" color="text.secondary">
                                Expires: {new Date(previewData.cache_expires_at).toLocaleString()}
                              </Typography>
                            )}
                          </Box>
                          {previewData.columns && previewData.columns.length > 0 ? (
                            <TableContainer component={Paper} sx={{ maxHeight: 500 }}>
                              <Table stickyHeader size="small">
                                <TableHead>
                                  <TableRow>
                                    {previewData.columns.map((col, idx) => (
                                      <TableCell key={idx} sx={{ fontWeight: 'bold' }}>
                                        {col}
                                      </TableCell>
                                    ))}
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {previewData.rows && previewData.rows.length > 0 ? (
                                    previewData.rows.map((row, rowIdx) => (
                                      <TableRow key={rowIdx}>
                                        {row.map((cell, cellIdx) => (
                                          <TableCell key={cellIdx}>
                                            {cell !== null && cell !== undefined
                                              ? String(cell)
                                              : 'NULL'}
                                          </TableCell>
                                        ))}
                                      </TableRow>
                                    ))
                                  ) : (
                                    <TableRow>
                                      <TableCell
                                        colSpan={previewData.columns.length}
                                        align="center"
                                      >
                                        No data returned
                                      </TableCell>
                                    </TableRow>
                                  )}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          ) : (
                            <Alert severity="info">No columns returned from query</Alert>
                          )}
                        </Box>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          Click "Preview" or select a tool to view data preview.
                        </Typography>
                      )}
                    </>
                  )}
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Select a database tool from the list to preview query results (first 10 rows).
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Create DB Tool Dialog */}
      <Dialog
        open={openCreateDialog}
        onClose={() => {
          setOpenCreateDialog(false);
          resetForm();
        }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create Database Tool</DialogTitle>
        <DialogContent>
          {createMutation.isError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to create database tool. Please check your inputs and try again.
            </Alert>
          )}
          <Box sx={{ mt: 1 }}>
            <TextField
              fullWidth
              label="Name"
              value={createForm.name}
              onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
              sx={{ mb: 2 }}
              required
            />
            <TextField
              fullWidth
              label="Description"
              value={createForm.description}
              onChange={(e) => setCreateForm({ ...createForm, description: e.target.value })}
              multiline
              rows={2}
              sx={{ mb: 2 }}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Database Type</InputLabel>
              <Select
                value={createForm.db_type}
                label="Database Type"
                onChange={(e) => {
                  const newType = e.target.value;
                  setCreateForm({
                    ...createForm,
                    db_type: newType,
                    port: getDefaultPort(newType),
                  });
                }}
              >
                <MenuItem value="mysql">MySQL</MenuItem>
                <MenuItem value="sqlserver">SQL Server</MenuItem>
                <MenuItem value="sqlite">SQLite</MenuItem>
                <MenuItem value="mongodb">MongoDB</MenuItem>
              </Select>
            </FormControl>
            {createForm.db_type === 'sqlite' ? (
              <TextField
                fullWidth
                label="Database File Path"
                value={createForm.database || createForm.host}
                onChange={(e) => {
                  setCreateForm({ 
                    ...createForm, 
                    database: e.target.value,
                    host: e.target.value 
                  });
                }}
                sx={{ mb: 2 }}
                required
                helperText="Path to SQLite database file (e.g., /path/to/database.db or ./data.db)"
              />
            ) : (
              <>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} sm={8}>
                    <TextField
                      fullWidth
                      label="Host"
                      value={createForm.host}
                      onChange={(e) => setCreateForm({ ...createForm, host: e.target.value })}
                      required
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      fullWidth
                      label="Port"
                      type="number"
                      value={createForm.port}
                      onChange={(e) =>
                        setCreateForm({ ...createForm, port: parseInt(e.target.value) || getDefaultPort(createForm.db_type) })
                      }
                      required
                    />
                  </Grid>
                </Grid>
                <TextField
                  fullWidth
                  label="Database Name"
                  value={createForm.database}
                  onChange={(e) => setCreateForm({ ...createForm, database: e.target.value })}
                  sx={{ mb: 2 }}
                  required
                />
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Username"
                      value={createForm.username}
                      onChange={(e) => setCreateForm({ ...createForm, username: e.target.value })}
                      required
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Password"
                      type="password"
                      value={createForm.password}
                      onChange={(e) => setCreateForm({ ...createForm, password: e.target.value })}
                      required
                    />
                  </Grid>
                </Grid>
              </>
            )}
            <FormControlLabel
              control={
                <Switch
                  checked={createForm.allow_dynamic_sql}
                  onChange={(e) =>
                    setCreateForm({ ...createForm, allow_dynamic_sql: e.target.checked })
                  }
                />
              }
              label="Allow Dynamic SQL"
              sx={{ mb: 2 }}
            />
            {createForm.allow_dynamic_sql && (
              <TextField
                fullWidth
                label="Preset SQL Statement (Optional)"
                value={createForm.preset_sql_statement}
                onChange={(e) => setCreateForm({ ...createForm, preset_sql_statement: e.target.value })}
                multiline
                rows={4}
                sx={{ mb: 2 }}
                placeholder={
                  createForm.db_type === 'mongodb'
                    ? '{"collection": "users", "query": {}, "projection": {}, "limit": 1000}'
                    : 'SELECT id, name, email FROM users'
                }
                helperText="Base SQL statement. Dynamic input will be appended as WHERE condition or used as full SQL if this is empty."
              />
            )}
            <TextField
              fullWidth
              label={createForm.allow_dynamic_sql ? 'Default SQL Statement (Fallback)' : (createForm.db_type === 'mongodb' ? 'MongoDB Query (JSON)' : 'SQL Statement')}
              value={createForm.sql_statement}
              onChange={(e) => setCreateForm({ ...createForm, sql_statement: e.target.value })}
              multiline
              rows={createForm.allow_dynamic_sql ? 4 : 6}
              sx={{ mb: 2 }}
              required
              placeholder={
                createForm.db_type === 'mongodb'
                  ? '{"collection": "users", "query": {}, "projection": {}, "limit": 1000}'
                  : createForm.allow_dynamic_sql
                  ? 'SELECT * FROM table_name WHERE condition (used as fallback)'
                  : 'SELECT * FROM table_name WHERE condition'
              }
              helperText={
                createForm.allow_dynamic_sql
                  ? 'Default SQL statement used when dynamic SQL is disabled or no input provided'
                  : createForm.db_type === 'mongodb'
                  ? 'MongoDB query in JSON format'
                  : 'SQL SELECT statement to execute'
              }
            />
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Cache TTL (hours)"
                  type="number"
                  value={createForm.cache_ttl_hours}
                  onChange={(e) =>
                    setCreateForm({
                      ...createForm,
                      cache_ttl_hours: parseFloat(e.target.value) || 1.0,
                    })
                  }
                  inputProps={{ min: 0.1, max: 24, step: 0.1 }}
                  helperText="How long to cache query results (0.1-24 hours)"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={createForm.is_active}
                      onChange={(e) =>
                        setCreateForm({ ...createForm, is_active: e.target.checked })
                      }
                    />
                  }
                  label="Active"
                  sx={{ mt: 2 }}
                />
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenCreateDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleCreate}
            disabled={
              createMutation.isLoading ||
              !createForm.name.trim() ||
              !createForm.sql_statement.trim() ||
              (createForm.db_type === 'sqlite' 
                ? !createForm.database.trim() && !createForm.host.trim()
                : !createForm.host.trim() || 
                  !createForm.database.trim() || 
                  !createForm.username.trim() || 
                  !createForm.password.trim())
            }
          >
            {createMutation.isLoading ? 'Creating...' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit DB Tool Dialog */}
      <Dialog
        open={openEditDialog}
        onClose={() => {
          setOpenEditDialog(false);
          setEditingToolId(null);
          resetForm();
        }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Edit Database Tool</DialogTitle>
        <DialogContent>
          {updateMutation.isError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to update database tool. Please check your inputs and try again.
            </Alert>
          )}
          <Box sx={{ mt: 1 }}>
            <TextField
              fullWidth
              label="Name"
              value={createForm.name}
              onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
              sx={{ mb: 2 }}
              required
            />
            <TextField
              fullWidth
              label="Description"
              value={createForm.description}
              onChange={(e) => setCreateForm({ ...createForm, description: e.target.value })}
              multiline
              rows={2}
              sx={{ mb: 2 }}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Database Type</InputLabel>
              <Select
                value={createForm.db_type}
                label="Database Type"
                onChange={(e) => {
                  const newType = e.target.value;
                  setCreateForm({
                    ...createForm,
                    db_type: newType,
                    port: getDefaultPort(newType),
                  });
                }}
              >
                <MenuItem value="mysql">MySQL</MenuItem>
                <MenuItem value="sqlserver">SQL Server</MenuItem>
                <MenuItem value="sqlite">SQLite</MenuItem>
                <MenuItem value="mongodb">MongoDB</MenuItem>
              </Select>
            </FormControl>
            {createForm.db_type === 'sqlite' ? (
              <TextField
                fullWidth
                label="Database File Path"
                value={createForm.database || createForm.host}
                onChange={(e) => {
                  setCreateForm({ 
                    ...createForm, 
                    database: e.target.value,
                    host: e.target.value 
                  });
                }}
                sx={{ mb: 2 }}
                required
                helperText="Path to SQLite database file (e.g., /path/to/database.db or ./data.db)"
              />
            ) : (
              <>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} sm={8}>
                    <TextField
                      fullWidth
                      label="Host"
                      value={createForm.host}
                      onChange={(e) => setCreateForm({ ...createForm, host: e.target.value })}
                      required
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      fullWidth
                      label="Port"
                      type="number"
                      value={createForm.port}
                      onChange={(e) =>
                        setCreateForm({ ...createForm, port: parseInt(e.target.value) || getDefaultPort(createForm.db_type) })
                      }
                      required
                    />
                  </Grid>
                </Grid>
                <TextField
                  fullWidth
                  label="Database Name"
                  value={createForm.database}
                  onChange={(e) => setCreateForm({ ...createForm, database: e.target.value })}
                  sx={{ mb: 2 }}
                  required
                />
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Username"
                      value={createForm.username}
                      onChange={(e) => setCreateForm({ ...createForm, username: e.target.value })}
                      required
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Password"
                      type="password"
                      value={createForm.password}
                      onChange={(e) => setCreateForm({ ...createForm, password: e.target.value })}
                      helperText="Leave blank to keep existing password, or enter new password"
                    />
                  </Grid>
                </Grid>
              </>
            )}
            <FormControlLabel
              control={
                <Switch
                  checked={createForm.allow_dynamic_sql}
                  onChange={(e) =>
                    setCreateForm({ ...createForm, allow_dynamic_sql: e.target.checked })
                  }
                />
              }
              label="Allow Dynamic SQL"
              sx={{ mb: 2 }}
            />
            {createForm.allow_dynamic_sql && (
              <TextField
                fullWidth
                label="Preset SQL Statement (Optional)"
                value={createForm.preset_sql_statement}
                onChange={(e) => setCreateForm({ ...createForm, preset_sql_statement: e.target.value })}
                multiline
                rows={4}
                sx={{ mb: 2 }}
                placeholder={
                  createForm.db_type === 'mongodb'
                    ? '{"collection": "users", "query": {}, "projection": {}, "limit": 1000}'
                    : 'SELECT id, name, email FROM users'
                }
                helperText="Base SQL statement. Dynamic input will be appended as WHERE condition or used as full SQL if this is empty."
              />
            )}
            <TextField
              fullWidth
              label={createForm.allow_dynamic_sql ? 'Default SQL Statement (Fallback)' : (createForm.db_type === 'mongodb' ? 'MongoDB Query (JSON)' : 'SQL Statement')}
              value={createForm.sql_statement}
              onChange={(e) => setCreateForm({ ...createForm, sql_statement: e.target.value })}
              multiline
              rows={createForm.allow_dynamic_sql ? 4 : 6}
              sx={{ mb: 2 }}
              required
              placeholder={
                createForm.db_type === 'mongodb'
                  ? '{"collection": "users", "query": {}, "projection": {}, "limit": 1000}'
                  : createForm.allow_dynamic_sql
                  ? 'SELECT * FROM table_name WHERE condition (used as fallback)'
                  : 'SELECT * FROM table_name WHERE condition'
              }
              helperText={
                createForm.allow_dynamic_sql
                  ? 'Default SQL statement used when dynamic SQL is disabled or no input provided'
                  : createForm.db_type === 'mongodb'
                  ? 'MongoDB query in JSON format'
                  : 'SQL SELECT statement to execute'
              }
            />
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Cache TTL (hours)"
                  type="number"
                  value={createForm.cache_ttl_hours}
                  onChange={(e) =>
                    setCreateForm({
                      ...createForm,
                      cache_ttl_hours: parseFloat(e.target.value) || 1.0,
                    })
                  }
                  inputProps={{ min: 0.1, max: 24, step: 0.1 }}
                  helperText="How long to cache query results (0.1-24 hours)"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={createForm.is_active}
                      onChange={(e) =>
                        setCreateForm({ ...createForm, is_active: e.target.checked })
                      }
                    />
                  }
                  label="Active"
                  sx={{ mt: 2 }}
                />
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenEditDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleUpdate}
            disabled={
              updateMutation.isLoading ||
              !createForm.name.trim() ||
              !createForm.sql_statement.trim() ||
              (createForm.db_type === 'sqlite' 
                ? !createForm.database.trim() && !createForm.host.trim()
                : !createForm.host.trim() || 
                  !createForm.database.trim() || 
                  !createForm.username.trim())
            }
          >
            {updateMutation.isLoading ? 'Updating...' : 'Update'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DBTools;