import React from 'react';
import {
    Box,
    Typography,
    Paper,
    Grid,
    Card,
    CardContent,
    Chip,
    LinearProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Divider,
    Alert
} from '@mui/material';
import {
    Psychology,
    Speed,
    TrendingUp,
    Schedule,
    CheckCircle,
    Info,
    ExpandMore,
    Gavel,
    Mic,
    TextFields
} from '@mui/icons-material';
import { AnalysisResult } from '../types';

interface ResultsDisplayProps {
    results: AnalysisResult;
    addDebugLog: (message: string) => void;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, addDebugLog }) => {
    React.useEffect(() => {
        addDebugLog(`Displaying results for ${results.input_type} analysis`);
    }, [results, addDebugLog]);

    const formatTimestamp = (timestamp: string) => {
        return new Date(timestamp).toLocaleString();
    };

    const formatConfidence = (confidence: number) => {
        return `${(confidence * 100).toFixed(1)}%`;
    };

    const getConfidenceColor = (confidence: number) => {
        if (confidence >= 0.8) return 'success';
        if (confidence >= 0.6) return 'warning';
        return 'error';
    };

    const getConfidenceLabel = (confidence: number) => {
        if (confidence >= 0.8) return 'High';
        if (confidence >= 0.6) return 'Medium';
        return 'Low';
    };

    return (
        <Box>
            {/* <Typography variant="h5" gutterBottom>
                Analysis Results
            </Typography> */}

            {/* Input Summary */}
            {/* <Card sx={{ mb: 3 }}>
                <CardContent>
                    {/* <Typography variant="h6" gutterBottom>
                        Analysis Summary
                    </Typography> */}
            {/* <Grid container spacing={2}>
                        <Grid item xs={6}>
                            <Typography variant="body2" color="text.secondary">
                                Input Type: {results.input_type}
                            </Typography>
                        </Grid>
                        <Grid item xs={6} style={{ textAlign: 'right' }}>
                            <Typography variant="body2" color="text.secondary">
                                Analysis Mode: {results.analysis_mode}
                            </Typography>
                        </Grid>
                    </Grid>
                </CardContent>
            </Card> */}


            {/* Best Model Results */}
            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        Predicted IPC Sections:
                    </Typography>
                    {results.results.best_predictions && results.results.best_predictions.length > 0 ? (
                        <Grid container spacing={2}>
                            {results.results.best_predictions.slice(0, 6).map((prediction, index) => (
                                <Grid item xs={12} sm={6} key={index}>
                                    <Box sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                                        <Typography variant="subtitle1" fontWeight="bold">
                                            {prediction.section}
                                        </Typography>
                                    </Box>
                                </Grid>
                            ))}
                        </Grid>
                    ) :
                        <Typography variant="body2" color="text.secondary">
                            No predictions found
                        </Typography>
                    }
                </CardContent>
            </Card>

            {/* All Model Results
            {results.results.all_model_results && results.results.all_model_results.length > 0 && (
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            All Model Results
                        </Typography>

                        {results.results.all_model_results.map((modelResult, index) => (
                            <Box key={index} sx={{ mb: 2 }}>
                                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                                    {modelResult.model_name}
                                </Typography>

                                {modelResult.predictions && modelResult.predictions.length > 0 ? (
                                    <Box>
                                        {modelResult.predictions.slice(0, 3).map((prediction, predIndex) => (
                                            <Box key={predIndex} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                                <Typography variant="body2">
                                                    {prediction.section}
                                                </Typography>
                                                <Chip
                                                    label={formatConfidence(prediction.confidence)}
                                                    color={getConfidenceColor(prediction.confidence) as any}
                                                    size="small"
                                                />
                                            </Box>
                                        ))}
                                        <Typography variant="body2" color="text.secondary">
                                            Confidence: {formatConfidence(modelResult.confidence_score)}
                                        </Typography>
                                    </Box>
                                ) : (
                                    <Typography variant="body2" color="text.secondary">
                                        No predictions
                                    </Typography>
                                )}

                                {index < results.results.all_model_results.length - 1 && <Divider sx={{ mt: 2 }} />}
                            </Box>
                        ))}
                    </CardContent>
                </Card>
            )} */}
        </Box>
    );
};

export default ResultsDisplay; 