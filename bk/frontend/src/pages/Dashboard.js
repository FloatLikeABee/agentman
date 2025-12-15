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
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import api from '../services/api';

const Dashboard = () => {
  const { data: status, isLoading, error } = useQuery('status', api.getStatus);
  const { data: collections } = useQuery('collections', api.getRAGCollections);
  const { data: agents } = useQuery('agents', api.getAgents);
  const { data: tools } = useQuery('tools', api.getTools);

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
        <Grid item xs={12} md={6}>
          <Card sx={{ boxShadow: 2 }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <RAGIcon color="primary" />
                Recent RAG Collections
              </Typography>
              <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
                {collections?.slice(0, 5).map((collection) => (
                  <Box key={collection.name} sx={{ mb: 2, p: 1, borderRadius: 1, bgcolor: 'grey.50' }}>
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

        <Grid item xs={12} md={6}>
          <Card sx={{ boxShadow: 2 }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AgentIcon color="secondary" />
                Active Agents
              </Typography>
              <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
                {agents?.slice(0, 5).map((agent) => (
                  <Box key={agent.id} sx={{ mb: 2, p: 1, borderRadius: 1, bgcolor: 'grey.50' }}>
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
      </Grid>
    </Box>
  );
};

export default Dashboard; 