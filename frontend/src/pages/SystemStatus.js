import React, { useState, useEffect } from 'react';
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
  Tabs,
  Tab,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Divider,
} from '@mui/material';
import {
  CheckCircle as ConnectedIcon,
  Error as DisconnectedIcon,
  Storage as ModelIcon,
  SmartToy as AgentIcon,
  Build as ToolIcon,
  Storage as CollectionIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import api from '../services/api';

const SystemStatus = () => {
  const [tab, setTab] = useState(0);
  const queryClient = useQueryClient();

  const { data: status, isLoading, error } = useQuery('status', api.getStatus, {
    refetchInterval: 10000, // Refetch every 10 seconds
  });
  const { data: mcpHosts = [] } = useQuery('mcp-hosts', api.getMCPHosts, {
    staleTime: 5 * 60 * 1000,
  });

  const {
    data: settingsData,
    isLoading: settingsLoading,
    error: settingsError,
  } = useQuery('system-settings', api.getSystemSettings);

  const updateSettingsMutation = useMutation(api.updateSystemSettings, {
    onSuccess: () => {
      queryClient.invalidateQueries('system-settings');
      queryClient.invalidateQueries('status');
    },
  });

  const settings = settingsData?.settings;
  const platformHasToken = settingsData?.platform_has_token || {};

  const [localSettings, setLocalSettings] = useState({
    default_llm_provider: '',
    default_model: '',
    providers_enabled: {},
    permissions: { allow_file_access: false, allow_shell_commands: false },
    external_credentials: {},
  });

  useEffect(() => {
    if (settings) {
      setLocalSettings({
        default_llm_provider: settings.default_llm_provider,
        default_model: settings.default_model,
        providers_enabled: settings.providers_enabled || {},
        permissions: settings.permissions || { allow_file_access: false, allow_shell_commands: false },
        external_credentials: settings.external_credentials || {},
      });
    }
  }, [settingsData]);

  const handleProviderToggle = (providerId) => {
    setLocalSettings((prev) => ({
      ...prev,
      providers_enabled: {
        ...prev.providers_enabled,
        [providerId]: !prev.providers_enabled?.[providerId],
      },
    }));
  };

  const handlePermissionToggle = (field) => {
    setLocalSettings((prev) => ({
      ...prev,
      permissions: {
        ...prev.permissions,
        [field]: !prev.permissions?.[field],
      },
    }));
  };

  const handleCredentialChange = (platform, field, value) => {
    setLocalSettings((prev) => ({
      ...prev,
      external_credentials: {
        ...prev.external_credentials,
        [platform]: {
          platform,
          username:
            field === 'username' ? value : prev.external_credentials?.[platform]?.username || '',
          access_token:
            field === 'access_token'
              ? value
              : prev.external_credentials?.[platform]?.access_token || '',
        },
      },
    }));
  };

  const handleSaveSettings = () => {
    const payload = {
      default_llm_provider: localSettings.default_llm_provider,
      default_model: localSettings.default_model,
      providers_enabled: localSettings.providers_enabled,
      permissions: localSettings.permissions,
      external_credentials: localSettings.external_credentials,
    };
    updateSettingsMutation.mutate(payload);
  };

  if (isLoading) {
    return <LinearProgress />;
  }

  if (error) {
    return <Alert severity="error">Failed to load system status</Alert>;
  }

  return (
    <Box sx={{ p: 1 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        System
      </Typography>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
        <Tab label="Status" />
        <Tab label="Settings" />
      </Tabs>

      {tab === 0 && (
        <>

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
            <Grid item xs={12} sm={3}>
              <Alert
                severity={status?.rag_collections?.length > 0 ? 'success' : 'info'}
                variant="outlined"
                sx={{ borderRadius: 2 }}
                icon={<CollectionIcon />}
              >
                <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                  Ground Control
                </Typography>
                <Typography variant="h6">
                  {status?.rag_collections?.length || 0} collections
                </Typography>
              </Alert>
            </Grid>
            <Grid item xs={12} sm={3}>
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
            <Grid item xs={12} sm={3}>
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
            <Grid item xs={12} sm={3}>
              <Alert
                severity={mcpHosts.length > 0 ? 'success' : 'info'}
                variant="outlined"
                sx={{ borderRadius: 2 }}
                icon={<ToolIcon />}
              >
                <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                  MCP Hosts
                </Typography>
                <Typography variant="h6">
                  {mcpHosts.length} configured
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
        </>
      )}

      {tab === 1 && (
        <Box sx={{ mt: 2 }}>
          {settingsLoading && <LinearProgress sx={{ mb: 2 }} />}
          {settingsError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to load system settings
            </Alert>
          )}
          {settings && (
            <>
              <Card sx={{ mb: 3, boxShadow: 2 }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    LLM Providers
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
                    <TextField
                      label="Default Provider"
                      size="small"
                      value={localSettings.default_llm_provider}
                      onChange={(e) =>
                        setLocalSettings((prev) => ({ ...prev, default_llm_provider: e.target.value }))
                      }
                    />
                    <TextField
                      label="Default Model"
                      size="small"
                      value={localSettings.default_model}
                      onChange={(e) =>
                        setLocalSettings((prev) => ({ ...prev, default_model: e.target.value }))
                      }
                    />
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {Object.keys(localSettings.providers_enabled || {}).map((pid) => (
                      <FormControlLabel
                        key={pid}
                        control={
                          <Switch
                            checked={!!localSettings.providers_enabled?.[pid]}
                            onChange={() => handleProviderToggle(pid)}
                          />
                        }
                        label={`Enable provider: ${pid}`}
                      />
                    ))}
                  </Box>
                </CardContent>
              </Card>

              <Card sx={{ mb: 3, boxShadow: 2 }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Ground Control Permissions
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                    These options control whether Ground Control is allowed to read/write files on the host
                    system and execute system commands. Enable with caution.
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={!!localSettings.permissions?.allow_file_access}
                        onChange={() => handlePermissionToggle('allow_file_access')}
                      />
                    }
                    label="Allow filesystem access (read/write)"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={!!localSettings.permissions?.allow_shell_commands}
                        onChange={() => handlePermissionToggle('allow_shell_commands')}
                      />
                    }
                    label="Allow execution of system shell commands"
                  />
                </CardContent>
              </Card>

              <Card sx={{ boxShadow: 2 }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    External Platform Credentials
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Configure credentials for platforms like Reddit and others. Tokens are stored securely and
                    only their presence is shown here.
                  </Typography>

                  {['reddit', 'github', 'slack'].map((platform) => {
                    const cred = localSettings.external_credentials?.[platform] || {};
                    const hasToken = platformHasToken?.[platform];
                    return (
                      <Box key={platform} sx={{ mb: 2 }}>
                        <Typography variant="subtitle1" sx={{ textTransform: 'capitalize' }}>
                          {platform}
                        </Typography>
                        <Grid container spacing={2} sx={{ mt: 0.5 }}>
                          <Grid item xs={12} sm={4}>
                            <TextField
                              label="Username / Client Id"
                              size="small"
                              fullWidth
                              value={cred.username || ''}
                              onChange={(e) =>
                                handleCredentialChange(platform, 'username', e.target.value)
                              }
                            />
                          </Grid>
                          <Grid item xs={12} sm={8}>
                            <TextField
                              label={
                                hasToken
                                  ? 'Access Token (leave blank to keep existing, set a value then clear to remove)'
                                  : 'Access Token / Secret'
                              }
                              size="small"
                              type="password"
                              fullWidth
                              value={cred.access_token === '***' ? '' : cred.access_token || ''}
                              onChange={(e) =>
                                handleCredentialChange(platform, 'access_token', e.target.value)
                              }
                            />
                          </Grid>
                        </Grid>
                        <Divider sx={{ mt: 2 }} />
                      </Box>
                    );
                  })}

                  <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
                    {updateSettingsMutation.isError && (
                      <Alert severity="error" sx={{ mr: 'auto' }}>
                        Failed to update settings. Please try again.
                      </Alert>
                    )}
                    <Button
                      variant="contained"
                      onClick={handleSaveSettings}
                      disabled={updateSettingsMutation.isLoading}
                    >
                      {updateSettingsMutation.isLoading ? 'Saving...' : 'Save Settings'}
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </>
          )}
        </Box>
      )}
    </Box>
  );
};

export default SystemStatus; 