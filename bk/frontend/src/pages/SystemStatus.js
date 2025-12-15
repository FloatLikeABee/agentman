import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Alert,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  CheckCircle as ConnectedIcon,
  Error as DisconnectedIcon,
  Storage as ModelIcon,
  SmartToy as AgentIcon,
  Build as ToolIcon,
  Storage as CollectionIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import api from '../services/api';

const SystemStatus = () => {
  const { data: status, isLoading, error } = useQuery('status', api.getStatus, {
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  if (isLoading) {
    return <LinearProgress />;
  }

  if (error) {
    return <Alert severity="error">Failed to load system status</Alert>;
  }

  return (
    <Box sx={{ p: 1 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        System Status
      </Typography>

      {/* Available Models */}
      <Card sx={{ mb: 4, boxShadow: 2 }}>
        <CardContent sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ModelIcon color="primary" />
            Available Models
          </Typography>
          <Grid container spacing={2}>
            {status?.available_models?.map((model) => (
              <Grid item key={model.name}>
                <Chip
                  icon={<ModelIcon />}
                  label={model.name}
                  variant="outlined"
                  size="small"
                  color="primary"
                  sx={{ fontWeight: 'medium' }}
                />
              </Grid>
            ))}
            {(!status?.available_models || status.available_models.length === 0) && (
              <Grid item>
                <Typography variant="body2" color="text.secondary">
                  No models available
                </Typography>
              </Grid>
            )}
          </Grid>
        </CardContent>
      </Card>

      {/* System Components */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ boxShadow: 2, transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CollectionIcon color="secondary" />
                RAG Collections ({status?.rag_collections?.length || 0})
              </Typography>
              <List dense sx={{ maxHeight: 200, overflowY: 'auto' }}>
                {status?.rag_collections?.map((collection) => (
                  <ListItem key={collection} sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 36 }}>
                      <CollectionIcon color="secondary" />
                    </ListItemIcon>
                    <ListItemText
                      primary={collection}
                      primaryTypographyProps={{ variant: 'body2', fontWeight: 'medium' }}
                    />
                  </ListItem>
                ))}
                {(!status?.rag_collections || status.rag_collections.length === 0) && (
                  <ListItem sx={{ px: 0 }}>
                    <ListItemText
                      primary="No collections"
                      primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }}
                    />
                  </ListItem>
                )}
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ boxShadow: 2, transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AgentIcon color="success" />
                Active Agents ({status?.active_agents?.length || 0})
              </Typography>
              <List dense sx={{ maxHeight: 200, overflowY: 'auto' }}>
                {status?.active_agents?.map((agent) => (
                  <ListItem key={agent} sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 36 }}>
                      <AgentIcon color="success" />
                    </ListItemIcon>
                    <ListItemText
                      primary={agent}
                      primaryTypographyProps={{ variant: 'body2', fontWeight: 'medium' }}
                    />
                  </ListItem>
                ))}
                {(!status?.active_agents || status.active_agents.length === 0) && (
                  <ListItem sx={{ px: 0 }}>
                    <ListItemText
                      primary="No active agents"
                      primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }}
                    />
                  </ListItem>
                )}
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ boxShadow: 2, transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ToolIcon color="warning" />
                Available Tools ({status?.active_tools?.length || 0})
              </Typography>
              <List dense sx={{ maxHeight: 200, overflowY: 'auto' }}>
                {status?.active_tools?.map((tool) => (
                  <ListItem key={tool} sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 36 }}>
                      <ToolIcon color="warning" />
                    </ListItemIcon>
                    <ListItemText
                      primary={tool}
                      primaryTypographyProps={{ variant: 'body2', fontWeight: 'medium' }}
                    />
                  </ListItem>
                ))}
                {(!status?.active_tools || status.active_tools.length === 0) && (
                  <ListItem sx={{ px: 0 }}>
                    <ListItemText
                      primary="No tools available"
                      primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }}
                    />
                  </ListItem>
                )}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* System Health */}
      <Card sx={{ boxShadow: 2 }}>
        <CardContent sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ConnectedIcon color="info" />
            System Health Overview
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}>
              <Alert
                severity={status?.rag_collections?.length > 0 ? 'success' : 'info'}
                variant="outlined"
                sx={{ borderRadius: 2 }}
                icon={<CollectionIcon />}
              >
                <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                  RAG System
                </Typography>
                <Typography variant="h6">
                  {status?.rag_collections?.length || 0} collections
                </Typography>
              </Alert>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Alert
                severity={status?.active_agents?.length > 0 ? 'success' : 'info'}
                variant="outlined"
                sx={{ borderRadius: 2 }}
                icon={<AgentIcon />}
              >
                <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                  Agent System
                </Typography>
                <Typography variant="h6">
                  {status?.active_agents?.length || 0} agents
                </Typography>
              </Alert>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Alert
                severity={status?.active_tools?.length > 0 ? 'success' : 'info'}
                variant="outlined"
                sx={{ borderRadius: 2 }}
                icon={<ToolIcon />}
              >
                <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                  Tool System
                </Typography>
                <Typography variant="h6">
                  {status?.active_tools?.length || 0} tools
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default SystemStatus; 