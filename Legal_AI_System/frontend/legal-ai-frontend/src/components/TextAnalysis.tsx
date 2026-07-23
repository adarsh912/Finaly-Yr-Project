import React, { useState } from 'react';
import {
    Box,
    TextField,
    Button,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Typography,
    Alert
} from '@mui/material';
import { AnalysisResult } from '../types';
import { apiService, handleApiError } from '../services/api';

interface TextAnalysisProps {
    onAnalysis: (result: AnalysisResult) => void;
    onError: (error: string) => void;
    onLoading: (loading: boolean) => void;
    addDebugLog: (message: string) => void;
}

const TextAnalysis: React.FC<TextAnalysisProps> = ({
    onAnalysis,
    onError,
    onLoading,
    addDebugLog
}) => {
    const [text, setText] = useState('');
    const [mode, setMode] = useState<'single' | 'multimodal'>('multimodal');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!text.trim()) {
            onError('Please enter some text to analyze');
            return;
        }

        addDebugLog(`Submitting text analysis: ${text.substring(0, 50)}... (${mode} mode)`);
        onLoading(true);

        try {
            const response = await apiService.analyzeText(text, mode);
            addDebugLog(`Text analysis API call successful`);
            onAnalysis(response);
        } catch (err) {
            const errorMessage = handleApiError(err);
            addDebugLog(`Text analysis API call failed: ${errorMessage}`);
            onError(errorMessage);
        }
    };

    return (
        <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
            {/* Analysis Mode selection hidden for now */}
            {/*
            <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Analysis Mode</InputLabel>
                <Select
                    value={mode}
                    label="Analysis Mode"
                    onChange={(e) => {
                        const newMode = e.target.value as 'single' | 'multimodal';
                        setMode(newMode);
                        addDebugLog(`Switched to ${newMode} mode`);
                    }}
                >
                    <MenuItem value="single">Single Model</MenuItem>
                    <MenuItem value="multimodal">Multimodal</MenuItem>
                </Select>
            </FormControl>
            */}

            <TextField
                fullWidth
                multiline
                rows={6}
                variant="outlined"
                label="Enter legal text to analyze"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter legal text here... (e.g., 'The accused committed theft by taking property without permission')"
                sx={{ mb: 2 }}
            />

            <Button
                type="submit"
                variant="contained"
                color="primary"
                fullWidth
                size="large"
                disabled={!text.trim()}
            >
                Analyze Text
            </Button>
        </Box>
    );
};

export default TextAnalysis; 