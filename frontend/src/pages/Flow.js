import React, { useState, useRef, useEffect } from 'react';
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
  CheckCircle as CheckCircleIcon,
  Send as SendIcon,
  Chat as ChatIcon,
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
  const { data: dialogues = [] } = useQuery('dialogues', api.getDialogues);

  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openExecuteDialog, setOpenExecuteDialog] = useState(false);
  const [editingFlowId, setEditingFlowId] = useState(null);
  const [selectedFlow, setSelectedFlow] = useState(null);
  const [executeResult, setExecuteResult] = useState(null);
  const [executeLoading, setExecuteLoading] = useState(false);
  // Dialogue conversation state for flow execution
  const [dialogueConversation, setDialogueConversation] = useState(null);
  const [dialogueMessage, setDialogueMessage] = useState('');
  const [dialogueLoading, setDialogueLoading] = useState(false);
  const [openDialogueDialog, setOpenDialogueDialog] = useState(false);
  // Store paused flow state for resuming
  const [pausedFlowState, setPausedFlowState] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    steps: [],
    is_active: true,
  });
  const [doneSteps, setDoneSteps] = useState(new Set());
  // Use ref to track latest formData for event handlers
  const formDataRef = useRef(formData);
  
  // Keep ref in sync with state
  useEffect(() => {
    formDataRef.current = formData;
  }, [formData]);

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
    setDoneSteps(new Set());
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

  const handleAddStep = (markAsDone = false) => {
    if (!currentStep.step_id || !currentStep.step_name || !currentStep.resource_id) {
      alert('Please fill in step ID, name, and resource ID');
      return;
    }
    // Use functional update to ensure we have the latest state
    setFormData((prevFormData) => {
      const newSteps = [...prevFormData.steps, { ...currentStep }];
      const newIndex = newSteps.length - 1;
      console.log('=== ADDING STEP ===');
      console.log('Previous steps count:', prevFormData.steps.length);
      console.log('Previous steps:', prevFormData.steps);
      console.log('New step to add:', currentStep);
      console.log('New steps count:', newSteps.length);
      console.log('All steps after add:', newSteps);
      const newFormData = {
        ...prevFormData,
        steps: newSteps,
      };
      console.log('New formData:', newFormData);
      // Immediately update the ref to ensure it's in sync
      formDataRef.current = newFormData;
      console.log('Updated formDataRef.current:', formDataRef.current);
      console.log('=== END ADDING STEP ===');
      
      // Mark as done if requested
      if (markAsDone) {
        setDoneSteps((prev) => new Set([...prev, newIndex]));
      }
      
      return newFormData;
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
    // Remove from done steps if it was done
    setDoneSteps((prev) => {
      const newSet = new Set(prev);
      newSet.delete(index);
      // Adjust indices for steps after the removed one
      const adjustedSet = new Set();
      newSet.forEach((idx) => {
        if (idx < index) {
          adjustedSet.add(idx);
        } else if (idx > index) {
          adjustedSet.add(idx - 1);
        }
      });
      return adjustedSet;
    });
  };

  const handleStepDone = (index) => {
    // Mark step as done and ensure it's in the payload
    setDoneSteps((prev) => new Set([...prev, index]));
    // Force update to ensure step is in formData
    setFormData((prev) => {
      const updatedSteps = [...prev.steps];
      if (updatedSteps[index]) {
        // Step is already in the array, just mark it as done
        return { ...prev, steps: updatedSteps };
      }
      return prev;
    });
  };

  const handleCreateFlow = () => {
    console.log('=== CREATING/UPDATING FLOW ===');
    console.log('Current formData state:', formData);
    console.log('Current formDataRef.current:', formDataRef.current);
    console.log('Current formData.steps:', formData.steps);
    console.log('Current formDataRef.current.steps:', formDataRef.current.steps);
    console.log('Done steps:', Array.from(doneSteps));
    
    // Use ref to get the latest formData state, but also check the actual state
    const latestFormData = formDataRef.current;
    const stateFormData = formData;
    
    // Use whichever has more steps (should be the same, but this is a safety check)
    const dataToUse = latestFormData.steps.length >= stateFormData.steps.length ? latestFormData : stateFormData;
    
    console.log('Using formData with', dataToUse.steps.length, 'steps');
    
    if (!dataToUse.name || dataToUse.steps.length === 0) {
      alert('Please provide a name and at least one step');
      return;
    }

    // Include all steps in the payload
    // The "Done" button is a visual indicator that the step is finalized
    // All steps are included regardless of "done" status
    const stepsToInclude = dataToUse.steps;
    
    // Ensure we're using the latest formData state
    const payload = {
      name: dataToUse.name,
      description: dataToUse.description,
      steps: [...stepsToInclude], // Create a new array to ensure it's not a reference issue
      is_active: dataToUse.is_active,
    };
    console.log('Final payload:', payload);
    console.log('Payload steps count:', payload.steps.length);
    console.log('Payload steps:', JSON.stringify(payload.steps, null, 2));
    console.log('=== END CREATING/UPDATING FLOW ===');
    if (editingFlowId) {
      updateFlowMutation.mutate({ flowId: editingFlowId, payload });
    } else {
      createFlowMutation.mutate(payload);
    }
  };

  const handleEditFlow = (flow) => {
    const flowSteps = flow.steps || [];
    console.log('Editing flow:', flow);
    console.log('Flow steps:', flowSteps);
    console.log('Flow steps count:', flowSteps.length);
    setFormData({
      name: flow.name || '',
      description: flow.description || '',
      steps: flowSteps,
      is_active: flow.is_active !== undefined ? flow.is_active : true,
    });
    // Mark all existing steps as done when editing
    setDoneSteps(new Set(flowSteps.map((_, idx) => idx)));
    setEditingFlowId(flow.id);
    setOpenCreateDialog(true);
  };

  const handleExecuteFlow = async () => {
    if (!selectedFlow) return;
    setExecuteLoading(true);
    setExecuteResult(null);
    setDialogueConversation(null);
    try {
      const result = await api.executeFlow(selectedFlow.id, {
        initial_input: '',
        context: {},
      });
      setExecuteResult(result);
      
      // Check if flow is paused waiting for dialogue
      if (result.metadata?.paused && result.metadata?.waiting_for_dialogue) {
        // Store paused state for resuming
        setPausedFlowState({
          flowId: result.flow_id,
          pausedAtStep: result.metadata.paused_at_step,
          pausedStepId: result.metadata.paused_step_id,
          stepResults: result.step_results,
          dialogueConversationId: result.metadata.dialogue_conversation_id,
          dialogueProfileId: result.metadata.dialogue_profile_id,
        });
        
        // Find the dialogue step
        const dialogueStep = result.step_results?.find(
          step => step.step_type === 'dialogue' && step.success && step.metadata?.dialogue_response
        );
        if (dialogueStep) {
          const dialogueData = dialogueStep.metadata.dialogue_response;
          const waitingForInitial = dialogueData.metadata?.waiting_for_initial_message;
          setDialogueConversation({
            stepId: dialogueStep.step_id,
            stepName: dialogueStep.step_name,
            dialogueId: dialogueData.profile_id,
            conversationId: dialogueData.conversation_id,
            conversationHistory: dialogueData.conversation_history || [],
            isComplete: dialogueData.is_complete,
            needsMoreInfo: dialogueData.needs_more_info || waitingForInitial,
            turnNumber: dialogueData.turn_number,
            maxTurns: dialogueData.max_turns,
            waitingForInitial: waitingForInitial,
          });
          setOpenDialogueDialog(true);
        }
      } else {
        // Flow completed normally, clear paused state
        setPausedFlowState(null);
      }
    } catch (error) {
      setExecuteResult({
        success: false,
        error: error.response?.data?.detail || error.message,
      });
    } finally {
      setExecuteLoading(false);
    }
  };

  const handleResumeFlow = async () => {
    if (!pausedFlowState || !selectedFlow) return;
    
    setExecuteLoading(true);
    try {
      // Resume flow from the paused step
      const result = await api.executeFlow(pausedFlowState.flowId, {
        initial_input: '',
        context: {
          conversation_id: pausedFlowState.dialogueConversationId,
        },
        resume_from_step: pausedFlowState.pausedAtStep + 1, // Resume from next step
        previous_step_results: pausedFlowState.stepResults,
      });
      
      setExecuteResult(result);
      setPausedFlowState(null);
      
      // Check if flow completed or paused again
      if (result.metadata?.paused && result.metadata?.waiting_for_dialogue) {
        // Flow paused again at another dialogue step
        const dialogueStep = result.step_results?.find(
          step => step.step_type === 'dialogue' && step.success && step.metadata?.dialogue_response
        );
        if (dialogueStep) {
          const dialogueData = dialogueStep.metadata.dialogue_response;
          setDialogueConversation({
            stepId: dialogueStep.step_id,
            stepName: dialogueStep.step_name,
            dialogueId: dialogueData.profile_id,
            conversationId: dialogueData.conversation_id,
            conversationHistory: dialogueData.conversation_history || [],
            isComplete: dialogueData.is_complete,
            needsMoreInfo: dialogueData.needs_more_info,
            turnNumber: dialogueData.turn_number,
            maxTurns: dialogueData.max_turns,
            waitingForInitial: false,
          });
          setPausedFlowState({
            flowId: result.flow_id,
            pausedAtStep: result.metadata.paused_at_step,
            pausedStepId: result.metadata.paused_step_id,
            stepResults: result.step_results,
            dialogueConversationId: result.metadata.dialogue_conversation_id,
            dialogueProfileId: result.metadata.dialogue_profile_id,
          });
          setOpenDialogueDialog(true);
        }
      }
    } catch (error) {
      setExecuteResult({
        success: false,
        error: error.response?.data?.detail || error.message,
      });
    } finally {
      setExecuteLoading(false);
    }
  };

  const handleContinueDialogue = async () => {
    if (!dialogueConversation || !dialogueMessage.trim()) return;
    
    setDialogueLoading(true);
    try {
      let result;
      
      // If waiting for initial message or no conversation_id, start the dialogue
      if (dialogueConversation.waitingForInitial || !dialogueConversation.conversationId) {
        result = await api.startDialogue(dialogueConversation.dialogueId, {
          initial_message: dialogueMessage,
        });
      } else {
        // Continue existing conversation
        result = await api.continueDialogue(dialogueConversation.dialogueId, {
          conversation_id: dialogueConversation.conversationId,
          user_message: dialogueMessage,
        });
      }
      
      // Update conversation state
      setDialogueConversation({
        ...dialogueConversation,
        conversationId: result.conversation_id,
        conversationHistory: result.conversation_history || [],
        isComplete: result.is_complete,
        needsMoreInfo: result.needs_more_info,
        turnNumber: result.turn_number,
        waitingForInitial: false,  // No longer waiting after first message
      });
      setDialogueMessage('');
      
      // If dialogue is complete, automatically resume the flow
      if (result.is_complete && pausedFlowState) {
        setOpenDialogueDialog(false);
        // Update pausedFlowState with the latest conversation_id before resuming
        const updatedPausedState = {
          ...pausedFlowState,
          dialogueConversationId: result.conversation_id || pausedFlowState.dialogueConversationId,
        };
        setPausedFlowState(updatedPausedState);
        // Wait a moment for dialog to close, then resume flow with updated state
        setTimeout(async () => {
          if (!updatedPausedState || !selectedFlow) return;
          
          setExecuteLoading(true);
          try {
            // Resume flow from the paused step with updated conversation_id
            const resumeResult = await api.executeFlow(updatedPausedState.flowId, {
              initial_input: '',
              context: {
                conversation_id: updatedPausedState.dialogueConversationId,
              },
              resume_from_step: updatedPausedState.pausedAtStep + 1,
              previous_step_results: updatedPausedState.stepResults,
            });
            
            setExecuteResult(resumeResult);
            setPausedFlowState(null);
            
            // Check if flow completed or paused again
            if (resumeResult.metadata?.paused && resumeResult.metadata?.waiting_for_dialogue) {
              // Flow paused again at another dialogue step
              const dialogueStep = resumeResult.step_results?.find(
                step => step.step_type === 'dialogue' && step.success && step.metadata?.dialogue_response
              );
              if (dialogueStep) {
                const dialogueData = dialogueStep.metadata.dialogue_response;
                setDialogueConversation({
                  stepId: dialogueStep.step_id,
                  stepName: dialogueStep.step_name,
                  dialogueId: dialogueData.profile_id,
                  conversationId: dialogueData.conversation_id,
                  conversationHistory: dialogueData.conversation_history || [],
                  isComplete: dialogueData.is_complete,
                  needsMoreInfo: dialogueData.needs_more_info,
                  turnNumber: dialogueData.turn_number,
                  maxTurns: dialogueData.max_turns,
                  waitingForInitial: false,
                });
                setPausedFlowState({
                  flowId: resumeResult.flow_id,
                  pausedAtStep: resumeResult.metadata.paused_at_step,
                  pausedStepId: resumeResult.metadata.paused_step_id,
                  stepResults: resumeResult.step_results,
                  dialogueConversationId: resumeResult.metadata.dialogue_conversation_id,
                  dialogueProfileId: resumeResult.metadata.dialogue_profile_id,
                });
                setOpenDialogueDialog(true);
              }
            }
          } catch (error) {
            setExecuteResult({
              success: false,
              error: error.response?.data?.detail || error.message,
            });
          } finally {
            setExecuteLoading(false);
          }
        }, 500);
      }
    } catch (error) {
      alert(error.response?.data?.detail || error.message || 'Failed to continue dialogue');
    } finally {
      setDialogueLoading(false);
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
      case 'dialogue':
        return dialogues.map((d) => ({ value: d.id, label: d.name }));
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
              <Typography variant="h6" sx={{ mb: 2 }}>
                Steps ({formData.steps.length})
              </Typography>
              {formData.steps.length === 0 && (
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  No steps added yet. Add steps below.
                </Typography>
              )}
              {formData.steps.map((step, idx) => (
                <Accordion key={idx} sx={{ mb: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <Chip label={step.step_type} size="small" />
                      <Typography>{step.step_name}</Typography>
                      {doneSteps.has(idx) && (
                        <CheckCircleIcon fontSize="small" color="success" />
                      )}
                      <Box sx={{ ml: 'auto', display: 'flex', gap: 0.5 }}>
                        {!doneSteps.has(idx) && (
                          <Button
                            size="small"
                            variant="contained"
                            color="success"
                            startIcon={<CheckCircleIcon />}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStepDone(idx);
                            }}
                            sx={{ minWidth: 'auto', px: 1 }}
                          >
                            Done
                          </Button>
                        )}
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveStep(idx);
                          }}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Typography variant="body2">Resource: {step.resource_id}</Typography>
                      {step.use_previous_output && (
                        <Typography variant="body2" color="primary">
                          Uses previous step output
                        </Typography>
                      )}
                      {step.input_query && !step.use_previous_output && (
                        <Typography variant="body2">
                          Input Query: {step.input_query}
                        </Typography>
                      )}
                      {doneSteps.has(idx) && (
                        <Chip
                          label="Done"
                          size="small"
                          color="success"
                          icon={<CheckCircleIcon />}
                          sx={{ alignSelf: 'flex-start' }}
                        />
                      )}
                    </Box>
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
                      onChange={(e) => {
                        console.log('Step type changed to:', e.target.value);
                        setCurrentStep({ ...currentStep, step_type: e.target.value, resource_id: '' });
                      }}
                      label="Step Type"
                    >
                      <MenuItem value="customization">Customization</MenuItem>
                      <MenuItem value="agent">Agent</MenuItem>
                      <MenuItem value="db_tool">DB Tool</MenuItem>
                      <MenuItem value="request">Request</MenuItem>
                      <MenuItem value="crawler">Crawler</MenuItem>
                      <MenuItem value="dialogue">Dialogue</MenuItem>
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
                  <Box sx={{ mb: 2, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                    <Typography variant="body2">
                      Current step: {currentStep.step_id || '(no ID)'} | {currentStep.step_name || '(no name)'} | {currentStep.resource_id || '(no resource)'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Total steps in form: {formData.steps.length}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="outlined"
                      startIcon={<AddIcon />}
                      onClick={() => {
                        console.log('Add Step button clicked');
                        console.log('Current step state:', currentStep);
                        console.log('Current formData steps:', formData.steps);
                        handleAddStep(false);
                      }}
                      sx={{ flex: 1 }}
                    >
                      Add Step
                    </Button>
                    <Button
                      variant="contained"
                      color="success"
                      startIcon={<CheckCircleIcon />}
                      onClick={() => {
                        console.log('Done button clicked');
                        console.log('Current step state:', currentStep);
                        console.log('Current formData steps:', formData.steps);
                        handleAddStep(true);
                      }}
                      sx={{ flex: 1 }}
                    >
                      Done
                    </Button>
                  </Box>
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
                        {step.step_type === 'dialogue' && step.metadata?.dialogue_response ? (
                          <Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                              <ChatIcon color="primary" />
                              <Typography variant="subtitle2">
                                Dialogue Conversation
                              </Typography>
                              <Chip
                                size="small"
                                label={`Turn: ${step.metadata.dialogue_response.turn_number}/${step.metadata.dialogue_response.max_turns}`}
                                sx={{ ml: 'auto' }}
                              />
                            </Box>
                            <Box
                              sx={{
                                p: 2,
                                bgcolor: 'grey.50',
                                borderRadius: 1,
                                maxHeight: 300,
                                overflow: 'auto',
                                mb: 2,
                              }}
                            >
                              {step.metadata.dialogue_response.conversation_history?.map((msg, idx) => (
                                <Box
                                  key={idx}
                                  sx={{
                                    mb: 2,
                                    display: 'flex',
                                    justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                                  }}
                                >
                                  <Box
                                    sx={{
                                      p: 1.5,
                                      borderRadius: 2,
                                      maxWidth: '80%',
                                      bgcolor: msg.role === 'user' ? 'primary.main' : 'grey.200',
                                      color: msg.role === 'user' ? 'white' : 'text.primary',
                                    }}
                                  >
                                    <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 0.5 }}>
                                      {msg.role === 'user' ? 'You' : 'Assistant'}
                                    </Typography>
                                    <Typography variant="body2">{msg.content}</Typography>
                                  </Box>
                                </Box>
                              ))}
                            </Box>
                            <Alert severity="info" sx={{ mb: 1 }}>
                              {step.metadata.dialogue_response.is_complete
                                ? 'Dialogue completed'
                                : step.metadata.dialogue_response.needs_more_info
                                ? 'Dialogue needs more information'
                                : 'Dialogue in progress'}
                            </Alert>
                            <Button
                              variant="outlined"
                              startIcon={<ChatIcon />}
                              onClick={() => {
                                const dialogueData = step.metadata.dialogue_response;
                                const waitingForInitial = dialogueData.metadata?.waiting_for_initial_message;
                                setDialogueConversation({
                                  stepId: step.step_id,
                                  stepName: step.step_name,
                                  dialogueId: dialogueData.profile_id,
                                  conversationId: dialogueData.conversation_id,
                                  conversationHistory: dialogueData.conversation_history || [],
                                  isComplete: dialogueData.is_complete,
                                  needsMoreInfo: dialogueData.needs_more_info || waitingForInitial,
                                  turnNumber: dialogueData.turn_number,
                                  maxTurns: dialogueData.max_turns,
                                  waitingForInitial: waitingForInitial,
                                });
                                setOpenDialogueDialog(true);
                              }}
                              fullWidth
                            >
                              Open Conversation Window
                            </Button>
                          </Box>
                        ) : (
                          <>
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
                          </>
                        )}
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

      {/* Dialogue Conversation Dialog */}
      <Dialog
        open={openDialogueDialog}
        onClose={() => !dialogueLoading && setOpenDialogueDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ChatIcon color="primary" />
            <Typography>
              {dialogueConversation?.stepName || 'Dialogue Conversation'}
            </Typography>
            {dialogueConversation && (
              <Chip
                size="small"
                label={`Turn: ${dialogueConversation.turnNumber}/${dialogueConversation.maxTurns}`}
                sx={{ ml: 'auto' }}
              />
            )}
          </Box>
        </DialogTitle>
        <DialogContent>
          {dialogueLoading && <LinearProgress sx={{ mb: 2 }} />}
          {dialogueConversation && (
            <Box>
              {/* Conversation History */}
              <Box
                sx={{
                  p: 2,
                  bgcolor: 'grey.50',
                  borderRadius: 1,
                  maxHeight: 400,
                  overflow: 'auto',
                  mb: 2,
                }}
              >
                {dialogueConversation.conversationHistory.length === 0 ? (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                    Start a conversation by typing a message below
                  </Typography>
                ) : (
                  dialogueConversation.conversationHistory.map((msg, idx) => (
                    <Box
                      key={idx}
                      sx={{
                        mb: 2,
                        display: 'flex',
                        justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                      }}
                    >
                      <Box
                        sx={{
                          p: 1.5,
                          borderRadius: 2,
                          maxWidth: '80%',
                          bgcolor: msg.role === 'user' ? 'primary.main' : 'grey.200',
                          color: msg.role === 'user' ? 'white' : 'text.primary',
                        }}
                      >
                        <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 0.5 }}>
                          {msg.role === 'user' ? 'You' : 'Assistant'}
                        </Typography>
                        <Typography variant="body2">{msg.content}</Typography>
                      </Box>
                    </Box>
                  ))
                )}
              </Box>

              {/* Status Messages */}
              {dialogueConversation.isComplete && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  Conversation complete. You can close this window.
                </Alert>
              )}
              {dialogueConversation.needsMoreInfo && !dialogueConversation.isComplete && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  The assistant needs more information. Please respond below.
                </Alert>
              )}

              {/* Message Input */}
              {!dialogueConversation.isComplete && (
                <TextField
                  fullWidth
                  multiline
                  rows={3}
                  label={dialogueConversation.waitingForInitial ? "Start conversation" : "Your message"}
                  value={dialogueMessage}
                  onChange={(e) => setDialogueMessage(e.target.value)}
                  disabled={dialogueLoading}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleContinueDialogue();
                    }
                  }}
                  sx={{ mb: 2 }}
                />
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialogueDialog(false)} disabled={dialogueLoading}>
            Close
          </Button>
          {dialogueConversation && !dialogueConversation.isComplete && (
            <Button
              onClick={handleContinueDialogue}
              variant="contained"
              disabled={dialogueLoading || !dialogueMessage.trim()}
              startIcon={<SendIcon />}
            >
              {dialogueLoading ? 'Sending...' : 'Send'}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Flow;

