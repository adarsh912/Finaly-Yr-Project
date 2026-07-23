import React, { useState } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Alert,
  CircularProgress
} from '@mui/material';
import TextAnalysis from './components/TextAnalysis';
import AudioAnalysis from './components/AudioAnalysis';
import ResultsDisplay from './components/ResultsDisplay';
import { AnalysisResult } from './types';
import { apiService } from './services/api';

function App() {
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [debugLog, setDebugLog] = useState<string[]>([]);

  const addDebugLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}`;
    console.log(logEntry);
    setDebugLog(prev => [...prev.slice(-9), logEntry]); // Keep last 10 entries
  };

  // Connect debug logger to API service
  React.useEffect(() => {
    apiService.setDebugLogger(addDebugLog);
  }, []);

  const handleAnalysis = (result: AnalysisResult) => {
    addDebugLog(`Analysis completed: ${result.input_type} mode`);
    setResults(result);
    setLoading(false);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    addDebugLog(`Error: ${errorMessage}`);
    setError(errorMessage);
    setLoading(false);
  };

  const handleLoading = (isLoading: boolean) => {
    setLoading(isLoading);
    if (isLoading) {
      addDebugLog('Starting analysis...');
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center" color="primary">
        Legal AI Analysis System
      </Typography>

      {/* Debug Panel - Hidden */}
      {/* <Paper sx={{ p: 2, mb: 2, backgroundColor: '#f5f5f5' }}>
        <Typography variant="h6" gutterBottom>Debug Log</Typography>
        <Box sx={{ maxHeight: 150, overflow: 'auto', fontFamily: 'monospace', fontSize: '0.8rem' }}>
          {debugLog.map((log, index) => (
            <div key={index}>{log}</div>
          ))}
        </Box>
      </Paper> */}

      {/* Main Content */}
      <Grid container spacing={3} alignItems="stretch">
        {/* Left Side - Audio Analysis */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h5" gutterBottom color="primary">
              Audio Analysis
            </Typography>
            <Box sx={{ flex: 1 }}>
              <AudioAnalysis
                onAnalysis={handleAnalysis}
                onError={handleError}
                onLoading={handleLoading}
                addDebugLog={addDebugLog}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Right Side - Text Analysis */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h5" gutterBottom color="primary">
              Text Analysis
            </Typography>
            <Box sx={{ flex: 1 }}>
              <TextAnalysis
                onAnalysis={handleAnalysis}
                onError={handleError}
                onLoading={handleLoading}
                addDebugLog={addDebugLog}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Loading Indicator */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
          <CircularProgress />
          <Typography sx={{ ml: 2, alignSelf: 'center' }}>
            Analyzing...
          </Typography>
        </Box>
      )}

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {/* Results Display - Bottom */}
      {results && (
        <Paper sx={{ p: 3, mt: 3 }}>
          <Typography variant="h5" gutterBottom color="primary">
            Analysis Results
          </Typography>
          <ResultsDisplay results={results} addDebugLog={addDebugLog} />
        </Paper>
      )}
    </Container>
  );
}

export default App;
