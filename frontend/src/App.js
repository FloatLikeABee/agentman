import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Box, Container } from '@mui/material';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import RAGManager from './pages/RAGManager';
import AgentManager from './pages/AgentManager';
import ToolManager from './pages/ToolManager';
import SystemStatus from './pages/SystemStatus';
import Customizations from './pages/Customizations';
import Crawler from './pages/Crawler';

function App() {
  return (
    <Router>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Header />
        <Container component="main" sx={{ mt: 4, mb: 4, flex: 1, maxWidth: 'lg' }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/rag" element={<RAGManager />} />
            <Route path="/agents" element={<AgentManager />} />
            <Route path="/tools" element={<ToolManager />} />
            <Route path="/customizations" element={<Customizations />} />
            <Route path="/crawler" element={<Crawler />} />
            <Route path="/status" element={<SystemStatus />} />
          </Routes>
        </Container>
      </Box>
    </Router>
  );
}

export default App; 