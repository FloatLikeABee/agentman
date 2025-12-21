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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  Edit as EditIcon,
  ExpandMore as ExpandMoreIcon,
  ArrowForward as ArrowForwardIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';

const Flow = () => {
  const queryClient = useQueryClient();
  const { data: flows = [], isLoading } = useQuery('flows', api.getFlows);
  const { data: customizations = [] } = useQuery('customizations', api.getCustomizations);
  const { data: agents = [] } = useQuery('agents', api.getAgents);
  const { data: dbTools = [] } = useQuery('db-tools', api.getDBTools);
  const { data: requestTools = [] } = useQuery('request-tools', api.getRequestTools);

  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openExecuteDialog, setOpenExecuteDialog] = useState(false);
  const [editingFlowId, setEditingFlowId] = useState(null);
  const [selectedFlow, setSelectedFlow] = useState(null);
  const [executeResult, setExecuteResult] = useState(null);
  const [executeLoading, setExecuteLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    steps: [],
    is_active: true,
  });
  const [currentStep, setCurrentStep] = useState({
    step_id: '',
    step_type: 'customization',
    step_name: '',
    resource_id: '',
    input_query: '',
    use_previous_output: false,
    output_mapping: null,
  });

  const createFlowMutation = useMutation(api.createFlow, {
    onSuccess: () => {
      queryClient.invalidateQueries('flows');
      setOpenCreateDialog(false);
      resetForm();
    },
  });

  const updateFlowMutation = useMutation(
    ({ flowId, payload }) => api.updateFlow(flowId, payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('flows');
        setOpenCreateDialog(false);
        setEditingFlowId(null);
        resetForm();
      },
    }
  );

  const deleteFlowMutation = useMutation(api.deleteFlow, {
    onSuccess: () => {
      queryClient.invalidateQueries('flows');
    },
  });

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      steps: [],
      is_active: true,
    });
    setCurrentStep({
      step_id: '',
      step_type: 'customization',
      step_name: '',
      resource_id: '',
      input_query: '',
      use_previous_output: false,
      output_mapping: null,
    });
  };

  const handleAddStep = () => {
    if (!currentStep.step_id || !currentStep.step_name || !currentStep.resource_id) {
      alert('Please fill in step ID, name, and resource ID');
      return;
    }
    setFormData({
      ...formData,
      steps: [...formData.steps, { ...currentStep }],
    });
    setCurrentStep({
      step_id: '',
      step_type: 'customization',
      step_name: '',
      resource_id: '',
      input_query: '',
      use_previous_output: false,
      output_mapping: null,
    });
  };

  const handleRemoveStep = (index) => {
    const newSteps = formData.steps.filter((_, i) => i !== index);
    setFormData({ ...formData, steps: newSteps });
  };

  const handleCreateFlow = () => {
    if (!formData.name || formData.steps.length === 0) {
      alert('Please provide a name and at least one step');
      return;
    }
    if (editingFlowId) {
      updateFlowMutation.mutate({ flowId: editingFlowId, payload: formData });
    } else {
      createFlowMutation.mutate(formData);
    }
  };

  const handleEditFlow = (flow) => {
    setFormData({
      name: flow.name || '',
      description: flow.description || '',
      steps: flow.steps || [],
      is_active: flow.is_active !== undefined ? flow.is_active : true,
    });
    setEditingFlowId(flow.id);
    setOpenCreateDialog(true);
  };

  const handleExecuteFlow = async () => {
    if (!selectedFlow) return;
    setExecuteLoading(true);
    setExecuteResult(null);
    try {
      const result = await api.executeFlow(selectedFlow.id, {
        initial_input: '',
        context: {},
      });
      setExecuteResult(result);
    } catch (error) {
      setExecuteResult({
        success: false,
        error: error.response?.data?.detail || error.message,
      });
    } finally {
      setExecuteLoading(false);
    }
  };

  const getResourceOptions = (stepType) => {
    switch (stepType) {
      case 'customization':
        return customizations.map((c) => ({ value: c.id, label: c.name }));
      case 'agent':
        return agents.map((a) => ({ value: a.id, label: a.name }));
      case 'db_tool':
        return dbTools.map((d) => ({ value: d.id, label: d.name }));
      case 'request':
        return requestTools.map((r) => ({ value: r.id, label: r.name }));
      case 'crawler':
        return [{ value: 'crawler', label: 'Crawler Service' }];
      default:
        return [];
    }
  };

  if (isLoading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Flow Manager</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            resetForm();
            setEditingFlowId(null);
            setOpenCreateDialog(true);
          }}
        >
          Create Flow
        </Button>
      </Box>

      {/* Flows List */}
      <Grid container spacing={3}>
        {flows.map((flow) => (
          <Grid item xs={12} md={6} lg={4} key={flow.id}>
            <Card sx={{ boxShadow: 2, transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="h6" sx={{ mb: 1 }}>{flow.name}</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {flow.description}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                      <Chip
                        label={`${flow.steps?.length || 0} Steps`}
                        size="small"
                        color="primary"
                      />
                      <Chip
                        label={flow.is_active ? 'Active' : 'Inactive'}
                        size="small"
                        color={flow.is_active ? 'success' : 'default'}
                      />
                    </Box>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                      {flow.steps?.map((step, idx) => (
                        <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Chip label={step.step_type} size="small" variant="outlined" />
                          <Typography variant="body2">{step.step_name}</Typography>
                          {idx < flow.steps.length - 1 && <ArrowForwardIcon fontSize="small" />}
                        </Box>
                      ))}
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    <IconButton
                      size="small"
                      color="primary"
                      onClick={() => {
                        setSelectedFlow(flow);
                        setOpenExecuteDialog(true);
                      }}
                      title="Execute Flow"
                    >
                      <RunIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="secondary"
                      onClick={() => handleEditFlow(flow)}
                      title="Edit Flow"
                    >
                      <EditIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => deleteFlowMutation.mutate(flow.id)}
                      title="Delete Flow"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Create/Edit Flow Dialog */}
      <Dialog open={openCreateDialog} onClose={() => setOpenCreateDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>{editingFlowId ? 'Edit Flow' : 'Create Flow'}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Flow Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                multiline
                rows={2}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.is_active}
                    onChange={(e) => setFormData({ ...formData, is_active: e.target.checked })}
                  />
                }
                label="Active"
              />
            </Grid>
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6" sx={{ mb: 2 }}>Steps</Typography>
              {formData.steps.map((step, idx) => (
                <Accordion key={idx} sx={{ mb: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <Chip label={step.step_type} size="small" />
                      <Typography>{step.step_name}</Typography>
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveStep(idx);
                        }}
                        sx={{ ml: 'auto' }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="body2">Resource: {step.resource_id}</Typography>
                    {step.use_previous_output && (
                      <Typography variant="body2" color="primary">
                        Uses previous step output
                      </Typography>
                    )}
                  </AccordionDetails>
                </Accordion>
              ))}
            </Grid>
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6" sx={{ mb: 2 }}>Add Step</Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Step ID"
                    value={currentStep.step_id}
                    onChange={(e) => setCurrentStep({ ...currentStep, step_id: e.target.value })}
                    placeholder="step_1"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Step Name"
                    value={currentStep.step_name}
                    onChange={(e) => setCurrentStep({ ...currentStep, step_name: e.target.value })}
                    placeholder="Step 1: Customization"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Step Type</InputLabel>
                    <Select
                      value={currentStep.step_type}
                      onChange={(e) => setCurrentStep({ ...currentStep, step_type: e.target.value, resource_id: '' })}
                      label="Step Type"
                    >
                      <MenuItem value="customization">Customization</MenuItem>
                      <MenuItem value="agent">Agent</MenuItem>
                      <MenuItem value="db_tool">DB Tool</MenuItem>
                      <MenuItem value="request">Request</MenuItem>
                      <MenuItem value="crawler">Crawler</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Resource</InputLabel>
                    <Select
                      value={currentStep.resource_id}
                      onChange={(e) => setCurrentStep({ ...currentStep, resource_id: e.target.value })}
                      label="Resource"
                    >
                      {getResourceOptions(currentStep.step_type).map((opt) => (
                        <MenuItem key={opt.value} value={opt.value}>
                          {opt.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={currentStep.use_previous_output}
                        onChange={(e) =>
                          setCurrentStep({ ...currentStep, use_previous_output: e.target.checked })
                        }
                      />
                    }
                    label="Use Previous Step Output"
                  />
                </Grid>
                {!currentStep.use_previous_output && (
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Input Query (if not using previous output)"
                      value={currentStep.input_query}
                      onChange={(e) => setCurrentStep({ ...currentStep, input_query: e.target.value })}
                      multiline
                      rows={2}
                    />
                  </Grid>
                )}
                <Grid item xs={12}>
                  <Button
                    variant="outlined"
                    startIcon={<AddIcon />}
                    onClick={handleAddStep}
                    fullWidth
                  >
                    Add Step
                  </Button>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenCreateDialog(false)}>Cancel</Button>
          <Button
            onClick={handleCreateFlow}
            variant="contained"
            disabled={createFlowMutation.isLoading || updateFlowMutation.isLoading}
          >
            {editingFlowId
              ? updateFlowMutation.isLoading
                ? 'Updating...'
                : 'Update Flow'
              : createFlowMutation.isLoading
              ? 'Creating...'
              : 'Create Flow'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Execute Flow Dialog */}
      <Dialog open={openExecuteDialog} onClose={() => setOpenExecuteDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Execute Flow: {selectedFlow?.name}</DialogTitle>
        <DialogContent>
          {executeLoading && <LinearProgress sx={{ mb: 2 }} />}
          {executeResult && (
            <Box>
              <Alert severity={executeResult.success ? 'success' : 'error'} sx={{ mb: 2 }}>
                {executeResult.success ? 'Flow executed successfully' : executeResult.error}
              </Alert>
              {executeResult.step_results?.map((step, idx) => (
                <Accordion key={idx} sx={{ mb: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <Chip
                        label={step.step_type}
                        size="small"
                        color={step.success ? 'success' : 'error'}
                      />
                      <Typography>{step.step_name}</Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ ml: 'auto' }}>
                        {step.execution_time?.toFixed(2)}s
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    {step.success ? (
                      <Box>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          <strong>Output:</strong>
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{
                            p: 1,
                            bgcolor: 'grey.100',
                            borderRadius: 1,
                            whiteSpace: 'pre-wrap',
                            maxHeight: 200,
                            overflow: 'auto',
                          }}
                        >
                          {typeof step.output === 'object'
                            ? JSON.stringify(step.output, null, 2)
                            : step.output}
                        </Typography>
                      </Box>
                    ) : (
                      <Alert severity="error">{step.error}</Alert>
                    )}
                  </AccordionDetails>
                </Accordion>
              ))}
              {executeResult.final_output && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    Final Output
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      p: 2,
                      bgcolor: 'primary.light',
                      borderRadius: 1,
                      whiteSpace: 'pre-wrap',
                    }}
                  >
                    {typeof executeResult.final_output === 'object'
                      ? JSON.stringify(executeResult.final_output, null, 2)
                      : executeResult.final_output}
                  </Typography>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenExecuteDialog(false)}>Close</Button>
          <Button
            onClick={handleExecuteFlow}
            variant="contained"
            disabled={executeLoading}
            startIcon={<RunIcon />}
          >
            {executeLoading ? 'Executing...' : 'Execute Flow'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Flow;

