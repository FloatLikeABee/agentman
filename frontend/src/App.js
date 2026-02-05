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
import Gathering from './pages/Gathering';
import DBTools from './pages/DBTools';
import Dialogue from './pages/Dialogue';
import Flow from './pages/Flow';
import GraphicDocumentGenerator from './pages/GraphicDocumentGenerator';
import ImageGenerator from './pages/ImageGenerator';
import BrowserAutomation from './pages/BrowserAutomation';
import ImageReader from './pages/ImageReader';
import PDFReader from './pages/PDFReader';

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
              <Route path="/images" element={<ImageGenerator />} />
              <Route path="/image-reader" element={<ImageReader />} />
              <Route path="/pdf-reader" element={<PDFReader />} />
              <Route path="/browser-automation" element={<BrowserAutomation />} />
              <Route path="/customizations" element={<Customizations />} />
              <Route path="/crawler" element={<Crawler />} />
              <Route path="/gathering" element={<Gathering />} />
              <Route path="/db-tools" element={<DBTools />} />
              <Route path="/dialogue" element={<Dialogue />} />
              <Route path="/flow" element={<Flow />} />
              <Route path="/graphic-document" element={<GraphicDocumentGenerator />} />
              <Route path="/status" element={<SystemStatus />} />
            </Routes>
          </Container>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App; 