import React, { useState, useCallback, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Typography,
  IconButton,
  Chip,
  Paper,
  Grid,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Add as AddIcon,
  AccountTree as CustomizationIcon,
  SmartToy as AgentIcon,
  Storage as DbToolIcon,
  Http as RequestIcon,
  Web as CrawlerIcon,
  Chat as DialogueIcon,
} from '@mui/icons-material';

const stepTypeIcons = {
  customization: CustomizationIcon,
  agent: AgentIcon,
  db_tool: DbToolIcon,
  request: RequestIcon,
  crawler: CrawlerIcon,
  dialogue: DialogueIcon,
};

const getNodeColor = (stepType) => {
  const colors = {
    customization: '#1976d2',
    agent: '#2e7d32',
    db_tool: '#ed6c02',
    request: '#9c27b0',
    crawler: '#d32f2f',
    dialogue: '#0288d1',
  };
  return colors[stepType] || '#757575';
};

const getNodeIcon = (stepType) => {
  const Icon = stepTypeIcons[stepType] || CustomizationIcon;
  return Icon;
};


const GraphicalFlowEditor = ({
  steps = [],
  onStepsChange,
  customizations = [],
  agents = [],
  dbTools = [],
  requestTools = [],
  dialogues = [],
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [openNodeDialog, setOpenNodeDialog] = useState(false);
  const [editingNodeId, setEditingNodeId] = useState(null);
  const [nodeFormData, setNodeFormData] = useState({
    step_id: '',
    step_name: '',
    step_type: 'customization',
    resource_id: '',
    input_query: '',
    use_previous_output: false,
    output_mapping: null,
  });

  // Convert steps to nodes and edges
  const stepsToFlow = useCallback((stepsArray) => {
    if (!stepsArray || stepsArray.length === 0) {
      return { nodes: [], edges: [] };
    }

    const flowNodes = stepsArray.map((step, index) => ({
      id: step.step_id || `step_${index}`,
      type: 'custom',
      position: { x: 250 * index, y: 100 },
      data: {
        stepType: step.step_type,
        stepName: step.step_name,
        resourceId: step.resource_id,
        inputQuery: step.input_query,
        usePreviousOutput: step.use_previous_output,
        outputMapping: step.output_mapping,
        stepId: step.step_id || `step_${index}`,
      },
    }));

    const flowEdges = [];
    for (let i = 0; i < flowNodes.length - 1; i++) {
      flowEdges.push({
        id: `edge_${i}_${i + 1}`,
        source: flowNodes[i].id,
        target: flowNodes[i + 1].id,
        type: 'smoothstep',
        animated: true,
        markerEnd: {
          type: MarkerType.ArrowClosed,
        },
      });
    }

    return { nodes: flowNodes, edges: flowEdges };
  }, []);

  // Convert nodes and edges back to steps
  const flowToSteps = useCallback((flowNodes, flowEdges) => {
    // Create a map of node positions based on edges
    const nodeOrder = [];
    const edgeMap = new Map();
    flowEdges.forEach(edge => {
      edgeMap.set(edge.target, edge.source);
    });

    // Find the start node (node with no incoming edges)
    const allTargets = new Set(flowEdges.map(e => e.target));
    const startNode = flowNodes.find(n => !allTargets.has(n.id));

    // Traverse nodes in order
    let currentNode = startNode;
    while (currentNode) {
      nodeOrder.push(currentNode);
      const nextEdge = flowEdges.find(e => e.source === currentNode.id);
      currentNode = nextEdge ? flowNodes.find(n => n.id === nextEdge.target) : null;
    }

    // If there are nodes not connected, add them at the end
    flowNodes.forEach(node => {
      if (!nodeOrder.find(n => n.id === node.id)) {
        nodeOrder.push(node);
      }
    });

    // Convert to steps array
    return nodeOrder.map(node => ({
      step_id: node.data.stepId || node.id,
      step_type: node.data.stepType,
      step_name: node.data.stepName || '',
      resource_id: node.data.resourceId || '',
      input_query: node.data.inputQuery || '',
      use_previous_output: node.data.usePreviousOutput || false,
      output_mapping: node.data.outputMapping || null,
    }));
  }, []);

  // Track previous steps to detect external changes
  const prevStepsRef = React.useRef(steps);
  const isInternalUpdateRef = React.useRef(false);

  // Initialize nodes and edges from steps, and update when steps change externally
  React.useEffect(() => {
    // Skip if this is an internal update (from our own flowToSteps)
    if (isInternalUpdateRef.current) {
      isInternalUpdateRef.current = false;
      return;
    }

    // Check if steps changed externally (not from our own updates)
    const stepsChanged = JSON.stringify(prevStepsRef.current) !== JSON.stringify(steps);
    
    if (stepsChanged) {
      const { nodes: initialNodes, edges: initialEdges } = stepsToFlow(steps);
      setNodes(initialNodes);
      setEdges(initialEdges);
      prevStepsRef.current = steps;
    }
  }, [steps, stepsToFlow]);

  // Update steps when nodes or edges change (debounced to avoid too many updates)
  React.useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (nodes.length > 0) {
        const newSteps = flowToSteps(nodes, edges);
        // Check if steps actually changed
        if (JSON.stringify(newSteps) !== JSON.stringify(steps)) {
          isInternalUpdateRef.current = true;
          onStepsChange(newSteps);
        }
      } else if (nodes.length === 0 && steps.length > 0) {
        // If nodes are cleared, clear steps too
        isInternalUpdateRef.current = true;
        onStepsChange([]);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges]);

  const onConnect = useCallback(
    (params) => {
      setEdges((eds) => addEdge(params, eds));
    },
    [setEdges]
  );

  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
    setEditingNodeId(node.id);
    setNodeFormData({
      step_id: node.data.stepId || node.id,
      step_name: node.data.stepName || '',
      step_type: node.data.stepType || 'customization',
      resource_id: node.data.resourceId || '',
      input_query: node.data.inputQuery || '',
      use_previous_output: node.data.usePreviousOutput || false,
      output_mapping: node.data.outputMapping || null,
    });
    setOpenNodeDialog(true);
  }, []);

  const handleAddNode = () => {
    const newNodeId = `step_${Date.now()}`;
    const newNode = {
      id: newNodeId,
      type: 'custom',
      position: {
        x: nodes.length > 0 ? nodes[nodes.length - 1].position.x + 250 : 100,
        y: 100,
      },
      data: {
        stepType: 'customization',
        stepName: 'New Step',
        resourceId: '',
        inputQuery: '',
        usePreviousOutput: false,
        outputMapping: null,
        stepId: newNodeId,
      },
    };

    setNodes((nds) => [...nds, newNode]);

    // Connect to previous node if exists
    if (nodes.length > 0) {
      const lastNode = nodes[nodes.length - 1];
      const newEdge = {
        id: `edge_${lastNode.id}_${newNodeId}`,
        source: lastNode.id,
        target: newNodeId,
        type: 'smoothstep',
        animated: true,
        markerEnd: {
          type: MarkerType.ArrowClosed,
        },
      };
      setEdges((eds) => [...eds, newEdge]);
    }

    // Open dialog to configure the new node
    setSelectedNode(newNode);
    setEditingNodeId(newNodeId);
    setNodeFormData({
      step_id: newNodeId,
      step_name: 'New Step',
      step_type: 'customization',
      resource_id: '',
      input_query: '',
      use_previous_output: false,
      output_mapping: null,
    });
    setOpenNodeDialog(true);
  };

  const handleDeleteNode = useCallback((nodeId = null) => {
    const idToDelete = nodeId || selectedNode?.id;
    if (!idToDelete) return;

    setNodes((nds) => nds.filter((node) => node.id !== idToDelete));
    setEdges((eds) =>
      eds.filter(
        (edge) => edge.source !== idToDelete && edge.target !== idToDelete
      )
    );
    if (selectedNode?.id === idToDelete) {
      setSelectedNode(null);
      setOpenNodeDialog(false);
    }
  }, [selectedNode]);

  const handleEditNode = useCallback((nodeId) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    setSelectedNode(node);
    setEditingNodeId(node.id);
    setNodeFormData({
      step_id: node.data.stepId || node.id,
      step_name: node.data.stepName || '',
      step_type: node.data.stepType || 'customization',
      resource_id: node.data.resourceId || '',
      input_query: node.data.inputQuery || '',
      use_previous_output: node.data.usePreviousOutput || false,
      output_mapping: node.data.outputMapping || null,
    });
    setOpenNodeDialog(true);
  }, [nodes]);

  // Custom Node Component - proper React component
  const CustomNode = React.memo(({ data, selected, id }) => {
    const Icon = getNodeIcon(data.stepType);
    const color = getNodeColor(data.stepType);
    const [showActions, setShowActions] = React.useState(false);

    return (
      <Paper
        elevation={selected ? 8 : 2}
        sx={{
          padding: 2,
          minWidth: 200,
          border: `2px solid ${selected ? color : 'transparent'}`,
          borderRadius: 2,
          backgroundColor: 'white',
          cursor: 'pointer',
          position: 'relative',
          '&:hover': {
            boxShadow: 6,
          },
        }}
        onMouseEnter={() => setShowActions(true)}
        onMouseLeave={() => setShowActions(false)}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Icon sx={{ color, fontSize: 24 }} />
          <Typography variant="subtitle2" sx={{ fontWeight: 'bold', textTransform: 'capitalize', flex: 1 }}>
            {data.stepType.replace('_', ' ')}
          </Typography>
          {(showActions || selected) && (
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                e.preventDefault();
                if (window.confirm('Are you sure you want to delete this node?')) {
                  handleDeleteNode(id);
                }
              }}
              sx={{ 
                padding: 0.5,
                '&:hover': { backgroundColor: 'error.light', color: 'white' }
              }}
              title="Delete Node"
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          )}
        </Box>
        <Typography variant="body2" sx={{ mb: 1, fontWeight: 'bold' }}>
          {data.stepName || 'Unnamed Step'}
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
          <Chip
            label={data.resourceId || 'No Resource'}
            size="small"
            sx={{ fontSize: '0.7rem', height: 20 }}
          />
          {data.usePreviousOutput && (
            <Chip
              label="Uses Previous"
              size="small"
              color="primary"
              sx={{ fontSize: '0.7rem', height: 20 }}
            />
          )}
        </Box>
      </Paper>
    );
  });

  const handleSaveNode = () => {
    if (!editingNodeId) return;

    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === editingNodeId) {
            return {
              ...node,
              data: {
                ...node.data,
                stepId: nodeFormData.step_id,
                stepType: nodeFormData.step_type,
                stepName: nodeFormData.step_name,
                resourceId: nodeFormData.resource_id,
                inputQuery: nodeFormData.input_query,
                usePreviousOutput: nodeFormData.use_previous_output,
                outputMapping: nodeFormData.output_mapping,
              },
            };
        }
        return node;
      })
    );

    setOpenNodeDialog(false);
    setSelectedNode(null);
    setEditingNodeId(null);
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

  // Memoize nodeTypes to avoid recreating on every render
  const nodeTypes = useMemo(() => ({
    custom: CustomNode,
  }), []);

  return (
    <Box sx={{ width: '100%', height: '600px', position: 'relative' }}>
      <Box sx={{ position: 'absolute', top: 10, right: 10, zIndex: 1000 }}>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleAddNode}
          sx={{ mb: 1 }}
        >
          Add Node
        </Button>
      </Box>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        style={{ background: '#f5f5f5' }}
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>

      {/* Node Configuration Dialog */}
      <Dialog
        open={openNodeDialog}
        onClose={() => {
          setOpenNodeDialog(false);
          setSelectedNode(null);
          setEditingNodeId(null);
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {editingNodeId ? 'Edit Node' : 'Add Node'}
          {selectedNode && (
            <IconButton
              onClick={handleDeleteNode}
              color="error"
              sx={{ position: 'absolute', right: 8, top: 8 }}
            >
              <DeleteIcon />
            </IconButton>
          )}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Step ID"
                value={nodeFormData.step_id}
                onChange={(e) =>
                  setNodeFormData({ ...nodeFormData, step_id: e.target.value })
                }
                placeholder="step_1"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Step Name"
                value={nodeFormData.step_name}
                onChange={(e) =>
                  setNodeFormData({ ...nodeFormData, step_name: e.target.value })
                }
                placeholder="Step 1: Customization"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Step Type</InputLabel>
                <Select
                  value={nodeFormData.step_type}
                  onChange={(e) =>
                    setNodeFormData({
                      ...nodeFormData,
                      step_type: e.target.value,
                      resource_id: '',
                    })
                  }
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
                  value={nodeFormData.resource_id}
                  onChange={(e) =>
                    setNodeFormData({ ...nodeFormData, resource_id: e.target.value })
                  }
                  label="Resource"
                >
                  {getResourceOptions(nodeFormData.step_type).map((opt) => (
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
                    checked={nodeFormData.use_previous_output}
                    onChange={(e) =>
                      setNodeFormData({
                        ...nodeFormData,
                        use_previous_output: e.target.checked,
                      })
                    }
                  />
                }
                label="Use Previous Step Output"
              />
            </Grid>
            {!nodeFormData.use_previous_output && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Input Query (if not using previous output)"
                  value={nodeFormData.input_query}
                  onChange={(e) =>
                    setNodeFormData({ ...nodeFormData, input_query: e.target.value })
                  }
                  multiline
                  rows={3}
                />
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setOpenNodeDialog(false);
            setSelectedNode(null);
            setEditingNodeId(null);
          }}>
            Cancel
          </Button>
          <Button onClick={handleSaveNode} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default GraphicalFlowEditor;

