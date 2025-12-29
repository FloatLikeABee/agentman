import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, CssBaseline, Box, Container } from '@mui/material';
import theme from './theme';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import RAGManager from './pages/RAGManager';
import AgentManager from './pages/AgentManager';
import ToolManager from './pages/ToolManager';
import SystemStatus from './pages/SystemStatus';
import Customizations from './pages/Customizations';
import Crawler from './pages/Crawler';
import DBTools from './pages/DBTools';
import Dialogue from './pages/Dialogue';
import Flow from './pages/Flow';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          minHeight: '100vh',
          position: 'relative',
          zIndex: 1,
        }}>
          <Header />
          <Container component="main" sx={{ mt: 4, mb: 4, flex: 1, maxWidth: 'lg', position: 'relative', zIndex: 1 }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/rag" element={<RAGManager />} />
              <Route path="/agents" element={<AgentManager />} />
              <Route path="/tools" element={<ToolManager />} />
              <Route path="/customizations" element={<Customizations />} />
              <Route path="/crawler" element={<Crawler />} />
              <Route path="/db-tools" element={<DBTools />} />
              <Route path="/dialogue" element={<Dialogue />} />
              <Route path="/flow" element={<Flow />} />
              <Route path="/status" element={<SystemStatus />} />
            </Routes>
          </Container>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App; 