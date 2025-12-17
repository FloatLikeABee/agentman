import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Storage as RAGIcon,
  SmartToy as AgentIcon,
  Build as ToolIcon,
  Info as StatusIcon,
  Tune as CustomIcon,
  Menu as MenuIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useQuery } from 'react-query';
import api from '../services/api';

const Header = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [anchorEl, setAnchorEl] = useState(null);

  const { data: status } = useQuery('status', api.getStatus, {
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/rag', label: 'RAG Manager', icon: <RAGIcon /> },
    { path: '/agents', label: 'Agent Manager', icon: <AgentIcon /> },
    { path: '/tools', label: 'Tool Manager', icon: <ToolIcon /> },
    { path: '/customizations', label: 'Customizations', icon: <CustomIcon /> },
    { path: '/status', label: 'System Status', icon: <StatusIcon /> },
  ];

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNavigate = (path) => {
    navigate(path);
    handleMenuClose();
  };

  return (
    <AppBar position="static" sx={{ mb: 2 }}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          RAG System
        </Typography>

        {isMobile ? (
          <>
            <IconButton
              color="inherit"
              onClick={handleMenuOpen}
              edge="end"
            >
              <MenuIcon />
            </IconButton>
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleMenuClose}
              anchorOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
            >
              {navItems.map((item) => (
                <MenuItem
                  key={item.path}
                  onClick={() => handleNavigate(item.path)}
                  selected={location.pathname === item.path}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {item.icon}
                    {item.label}
                  </Box>
                </MenuItem>
              ))}
            </Menu>
          </>
        ) : (
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            {navItems.map((item) => (
              <Button
                key={item.path}
                color="inherit"
                startIcon={item.icon}
                onClick={() => navigate(item.path)}
                sx={{
                  backgroundColor: location.pathname === item.path ? 'rgba(255,255,255,0.1)' : 'transparent',
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.2)',
                  },
                }}
              >
                {item.label}
              </Button>
            ))}
          </Box>
        )}
      </Toolbar>
    </AppBar>
  );
};

export default Header; 