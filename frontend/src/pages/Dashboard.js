import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  Storage as RAGIcon,
  SmartToy as AgentIcon,
  Build as ToolIcon,
  CheckCircle as ConnectedIcon,
  Error as DisconnectedIcon,
  Tune as CustomizationIcon,
  Chat as DialogueIcon,
  AccountTree as FlowIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import api from '../services/api';

const Dashboard = () => {
  const { data: status, isLoading, error } = useQuery('status', api.getStatus, { staleTime: 5 * 60 * 1000 }); // Cache for 5 minutes
  const { data: collections } = useQuery('collections', api.getRAGCollections, { staleTime: 5 * 60 * 1000 });
  const { data: agents } = useQuery('agents', api.getAgents, { staleTime: 5 * 60 * 1000 });
  const { data: tools } = useQuery('tools', api.getTools, { staleTime: 5 * 60 * 1000 });
  const { data: customizations } = useQuery('customizations', api.getCustomizations, { staleTime: 5 * 60 * 1000 });
  const { data: dialogues } = useQuery('dialogues', api.getDialogues, { staleTime: 5 * 60 * 1000 });
  const { data: flows } = useQuery('flows', api.getFlows, { staleTime: 5 * 60 * 1000 });
  const { data: conversations } = useQuery('conversations', api.getConversations, { staleTime: 5 * 60 * 1000 });

  if (isLoading) {
    return <LinearProgress />;
  }

  if (error) {
    return <Alert severity="error">Failed to load dashboard data</Alert>;
  }

  const stats = [
    {
      title: 'RAG Collections',
      value: collections?.length || 0,
      icon: <RAGIcon />,
      color: 'primary',
    },
    {
      title: 'Active Agents',
      value: agents?.length || 0,
      icon: <AgentIcon />,
      color: 'secondary',
    },
    {
      title: 'Available Tools',
      value: tools?.length || 0,
      icon: <ToolIcon />,
      color: 'success',
    },
    {
      title: 'Customizations',
      value: customizations?.length || 0,
      icon: <CustomizationIcon />,
      color: 'info',
    },
    {
      title: 'Dialogues',
      value: dialogues?.length || 0,
      icon: <DialogueIcon />,
      color: 'warning',
    },
    {
      title: 'Flows',
      value: flows?.length || 0,
      icon: <FlowIcon />,
      color: 'error',
    },
    {
      title: 'Conversations',
      value: conversations?.length || 0,
      icon: <DialogueIcon />,
      color: 'primary',
    },
  ];

  return (
    <Box sx={{ p: 1 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Dashboard
      </Typography>

      {/* System Status */}
      <Card sx={{ mb: 4, boxShadow: 2 }}>
        <CardContent sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ConnectedIcon color="success" />
            System Status
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 2 }}>
            <Typography variant="body1" color="text.secondary">
              {status?.available_models?.length || 0} models available
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Statistics */}
      <Typography variant="h5" gutterBottom sx={{ mb: 2 }}>
        Overview
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {stats.map((stat) => (
          <Grid item xs={12} sm={6} md={4} key={stat.title}>
            <Card sx={{ minHeight: 120, boxShadow: 2, transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-4px)' } }}>
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Box sx={{ color: `${stat.color}.main`, fontSize: '2.5rem' }}>
                    {stat.icon}
                  </Box>
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="h3" component="div" sx={{ fontWeight: 'bold' }}>
                      {stat.value}
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mt: 0.5 }}>
                      {stat.title}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Recent Activity */}
      <Typography variant="h5" gutterBottom sx={{ mb: 2 }}>
        Recent Activity
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6} lg={4}>
          <Card sx={{ boxShadow: 2 }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <RAGIcon color="primary" />
                Recent RAG Collections
              </Typography>
              <Box sx={{ maxHeight: 200, overflowY: 'auto', '&::-webkit-scrollbar': { width: '8px' }, '&::-webkit-scrollbar-track': { bgcolor: 'background.default', borderRadius: '4px' }, '&::-webkit-scrollbar-thumb': { bgcolor: 'primary.main', bgcolorOpacity: 0.5, borderRadius: '4px', '&:hover': { bgcolor: 'primary.light' } } }}>
                {collections?.slice(0, 5).map((collection) => (
                  <Box key={collection.name} sx={{ mb: 2, p: 1, borderRadius: 1, bgcolor: 'background.paper', border: '1px solid', borderColor: 'primary.main', borderOpacity: 0.3 }}>
                    <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                      {collection.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {collection.count} documents
                    </Typography>
                  </Box>
                ))}
                {(!collections || collections.length === 0) && (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                    No collections yet
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6} lg={4}>
          <Card sx={{ boxShadow: 2 }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AgentIcon color="secondary" />
                Active Agents
              </Typography>
              <Box sx={{ maxHeight: 200, overflowY: 'auto', '&::-webkit-scrollbar': { width: '8px' }, '&::-webkit-scrollbar-track': { bgcolor: 'background.default', borderRadius: '4px' }, '&::-webkit-scrollbar-thumb': { bgcolor: 'primary.main', bgcolorOpacity: 0.5, borderRadius: '4px', '&:hover': { bgcolor: 'primary.light' } } }}>
                {agents?.slice(0, 5).map((agent) => (
                  <Box key={agent.id} sx={{ mb: 2, p: 1, borderRadius: 1, bgcolor: 'background.paper', border: '1px solid', borderColor: 'primary.main', borderOpacity: 0.3 }}>
                    <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                      {agent.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {agent.model_name}
                    </Typography>
                  </Box>
                ))}
                {(!agents || agents.length === 0) && (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                    No active agents
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6} lg={4}>
          <Card sx={{ boxShadow: 2 }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CustomizationIcon color="info" />
                Recent Customizations
              </Typography>
              <Box sx={{ maxHeight: 200, overflowY: 'auto', '&::-webkit-scrollbar': { width: '8px' }, '&::-webkit-scrollbar-track': { bgcolor: 'background.default', borderRadius: '4px' }, '&::-webkit-scrollbar-thumb': { bgcolor: 'primary.main', bgcolorOpacity: 0.5, borderRadius: '4px', '&:hover': { bgcolor: 'primary.light' } } }}>
                {customizations?.slice(0, 5).map((customization) => (
                  <Box key={customization.id} sx={{ mb: 2, p: 1, borderRadius: 1, bgcolor: 'background.paper', border: '1px solid', borderColor: 'primary.main', borderOpacity: 0.3 }}>
                    <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                      {customization.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {customization.llm_provider || 'N/A'}
                    </Typography>
                  </Box>
                ))}
                {(!customizations || customizations.length === 0) && (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                    No customizations yet
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6} lg={4}>
          <Card sx={{ boxShadow: 2 }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <DialogueIcon color="warning" />
                Recent Dialogues
              </Typography>
              <Box sx={{ maxHeight: 200, overflowY: 'auto', '&::-webkit-scrollbar': { width: '8px' }, '&::-webkit-scrollbar-track': { bgcolor: 'background.default', borderRadius: '4px' }, '&::-webkit-scrollbar-thumb': { bgcolor: 'primary.main', bgcolorOpacity: 0.5, borderRadius: '4px', '&:hover': { bgcolor: 'primary.light' } } }}>
                {dialogues?.slice(0, 5).map((dialogue) => (
                  <Box key={dialogue.id} sx={{ mb: 2, p: 1, borderRadius: 1, bgcolor: 'background.paper', border: '1px solid', borderColor: 'primary.main', borderOpacity: 0.3 }}>
                    <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                      {dialogue.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {dialogue.description || 'No description'}
                    </Typography>
                  </Box>
                ))}
                {(!dialogues || dialogues.length === 0) && (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                    No dialogues yet
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6} lg={4}>
          <Card sx={{ boxShadow: 2 }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <FlowIcon color="error" />
                Recent Flows
              </Typography>
              <Box sx={{ maxHeight: 200, overflowY: 'auto', '&::-webkit-scrollbar': { width: '8px' }, '&::-webkit-scrollbar-track': { bgcolor: 'background.default', borderRadius: '4px' }, '&::-webkit-scrollbar-thumb': { bgcolor: 'primary.main', bgcolorOpacity: 0.5, borderRadius: '4px', '&:hover': { bgcolor: 'primary.light' } } }}>
                {flows?.slice(0, 5).map((flow) => (
                  <Box key={flow.id} sx={{ mb: 2, p: 1, borderRadius: 1, bgcolor: 'background.paper', border: '1px solid', borderColor: 'primary.main', borderOpacity: 0.3 }}>
                    <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                      {flow.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {flow.description || 'No description'}
                    </Typography>
                  </Box>
                ))}
                {(!flows || flows.length === 0) && (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                    No flows yet
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 