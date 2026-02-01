import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
  Tooltip,
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
  Web as CrawlerIcon,
  TravelExplore as GatheringIcon,
  Dns as DBIcon,
  Chat as DialogueIcon,
  AccountTree as FlowIcon,
  Image as ImageIcon,
  Web as BrowserIcon,
  TextFields as TextIcon,
  PictureAsPdf as PDFIcon,
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

  useQuery('status', api.getStatus, {
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/agents', label: 'Agent', icon: <AgentIcon /> },
    { path: '/rag', label: 'RAG', icon: <RAGIcon /> },
    { path: '/tools', label: 'Tool', icon: <ToolIcon /> },
    { path: '/images', label: 'Images', icon: <ImageIcon /> },
    { path: '/image-reader', label: 'Image Reader', icon: <TextIcon /> },
    { path: '/pdf-reader', label: 'PDF Reader', icon: <PDFIcon /> },
    { path: '/browser-automation', label: 'Browser', icon: <BrowserIcon /> },
    { path: '/customizations', label: 'Customizations', icon: <CustomIcon /> },
    { path: '/db-tools', label: 'Database', icon: <DBIcon /> },
    { path: '/crawler', label: 'Crawler', icon: <CrawlerIcon /> },
    { path: '/gathering', label: 'Gathering', icon: <GatheringIcon /> },
    { path: '/dialogue', label: 'Dialogue', icon: <DialogueIcon /> },
    { path: '/flow', label: 'Flow', icon: <FlowIcon /> },
    { path: '/status', label: 'System', icon: <StatusIcon /> },
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
    <AppBar position="sticky" sx={{ top: 0, zIndex: 1300, mb: 2 }}>
      <Toolbar>
        <Typography 
          variant="h6" 
          component="div" 
          sx={{ 
            flexGrow: 1,
            fontFamily: '"Orbitron", "Roboto", sans-serif',
            fontWeight: 700,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            background: 'linear-gradient(135deg, #9d4edd 0%, #ff6b35 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            textShadow: '0 0 30px rgba(157, 78, 221, 0.5)',
          }}
        >
          Ground Control
        </Typography>

        {isMobile ? (
          <>
            <IconButton
              color="inherit"
              onClick={handleMenuOpen}
              edge="end"
              sx={{
                color: '#9d4edd',
                '&:hover': {
                  color: '#ff6b35',
                  backgroundColor: 'rgba(157, 78, 221, 0.1)',
                },
              }}
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
              PaperProps={{
                sx: {
                  background: 'linear-gradient(135deg, #0f0519 0%, #1a1a1a 100%)',
                  border: '1px solid rgba(157, 78, 221, 0.3)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.8), 0 0 20px rgba(157, 78, 221, 0.1)',
                },
              }}
            >
              {navItems.map((item) => (
                <MenuItem
                  key={item.path}
                  onClick={() => handleNavigate(item.path)}
                  selected={location.pathname === item.path}
                  sx={{
                    color: location.pathname === item.path ? '#9d4edd' : '#e0e0e0',
                    '&:hover': {
                      backgroundColor: 'rgba(157, 78, 221, 0.2)',
                    },
                    '&.Mui-selected': {
                      backgroundColor: 'rgba(157, 78, 221, 0.3)',
                      '&:hover': {
                        backgroundColor: 'rgba(157, 78, 221, 0.4)',
                      },
                    },
                  }}
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
          <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
            {navItems.map((item) => (
              <Tooltip key={item.path} title={item.label} arrow placement="bottom">
                <IconButton
                  color="inherit"
                  onClick={() => navigate(item.path)}
                  sx={{
                    color: location.pathname === item.path ? '#9d4edd' : '#b0b0b0',
                    backgroundColor: location.pathname === item.path 
                      ? 'rgba(157, 78, 221, 0.15)' 
                      : 'transparent',
                    border: location.pathname === item.path 
                      ? '1px solid rgba(157, 78, 221, 0.4)' 
                      : '1px solid transparent',
                    borderRadius: '8px',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      color: '#9d4edd',
                      backgroundColor: 'rgba(157, 78, 221, 0.2)',
                      border: '1px solid rgba(157, 78, 221, 0.5)',
                      boxShadow: '0 0 15px rgba(157, 78, 221, 0.3)',
                      transform: 'translateY(-2px)',
                    },
                  }}
                >
                  {item.icon}
                </IconButton>
              </Tooltip>
            ))}
          </Box>
        )}
      </Toolbar>
    </AppBar>
  );
};

export default Header; 