import React, { useState, useCallback } from 'react';
import {
    Box,
    Button,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Typography,
    Alert,
    LinearProgress
} from '@mui/material';
import { CloudUpload, Mic, CheckCircle, Error } from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { AnalysisResult } from '../types';
import { apiService, handleApiError } from '../services/api';

interface AudioAnalysisProps {
    onAnalysis: (result: AnalysisResult) => void;
    onError: (error: string) => void;
    onLoading: (loading: boolean) => void;
    addDebugLog: (message: string) => void;
}

const AudioAnalysis: React.FC<AudioAnalysisProps> = ({
    onAnalysis,
    onError,
    onLoading,
    addDebugLog
}) => {
    const [mode, setMode] = useState<'single' | 'multimodal'>('multimodal');
    const [uploadProgress, setUploadProgress] = useState(0);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [analysisStatus, setAnalysisStatus] = useState<'idle' | 'uploading' | 'analyzing' | 'done' | 'error'>('idle');
    const [statusMessage, setStatusMessage] = useState('');

    const handleFileAnalysis = async (file: File) => {
        addDebugLog(`Audio file selected: ${file.name} (${file.size} bytes)`);
        onLoading(true);
        setUploadProgress(0);
        setAnalysisStatus('uploading');
        setStatusMessage('Uploading audio file...');

        try {
            const formData = new FormData();
            formData.append('audio', file);
            formData.append('mode', mode);

            addDebugLog(`Uploading audio file for ${mode} analysis...`);

            setAnalysisStatus('analyzing');
            setStatusMessage('Analyzing audio content...');

            const response = await apiService.analyzeAudio(formData, (progress) => {
                setUploadProgress(progress);
                addDebugLog(`Upload progress: ${progress}%`);
                if (progress < 100) {
                    setStatusMessage(`Uploading: ${progress}%`);
                } else {
                    setStatusMessage('Processing audio...');
                }
            });

            addDebugLog(`Audio analysis API call successful`);
            setAnalysisStatus('done');
            setStatusMessage('Analysis completed successfully!');
            setUploadProgress(100);
            onAnalysis(response);

            // Reset status after 3 seconds
            setTimeout(() => {
                setAnalysisStatus('idle');
                setStatusMessage('');
                setUploadProgress(0);
            }, 3000);

        } catch (err) {
            const errorMessage = handleApiError(err);
            addDebugLog(`Audio analysis API call failed: ${errorMessage}`);
            setAnalysisStatus('error');
            setStatusMessage(`Analysis failed: ${errorMessage}`);
            onError(errorMessage);

            // Reset status after 5 seconds
            setTimeout(() => {
                setAnalysisStatus('idle');
                setStatusMessage('');
                setUploadProgress(0);
            }, 5000);
        }
    };

    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles.length === 0) return;
        const file = acceptedFiles[0];
        setSelectedFile(file);
        setAnalysisStatus('idle');
        setStatusMessage('File selected. Click "Analyze Audio" to start.');
        addDebugLog(`File dropped: ${file.name}`);
    }, [addDebugLog]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'audio/*': ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        },
        multiple: false
    });

    const handleAnalyzeButton = () => {
        if (selectedFile) {
            handleFileAnalysis(selectedFile);
        } else {
            onError('Please select an audio file first');
        }
    };

    const getStatusIcon = () => {
        switch (analysisStatus) {
            case 'done':
                return <CheckCircle sx={{ color: 'success.main', mr: 1 }} />;
            case 'error':
                return <Error sx={{ color: 'error.main', mr: 1 }} />;
            default:
                return null;
        }
    };

    const getButtonText = () => {
        switch (analysisStatus) {
            case 'uploading':
                return 'Uploading...';
            case 'analyzing':
                return 'Analyzing...';
            case 'done':
                return 'Analysis Complete';
            case 'error':
                return 'Retry Analysis';
            default:
                return 'Analyze Audio';
        }
    };

    const isButtonDisabled = () => {
        return !selectedFile || analysisStatus === 'uploading' || analysisStatus === 'analyzing';
    };

    return (
        <Box sx={{ mt: 2 }}>
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
            <Box
                {...getRootProps()}
                sx={{
                    border: '2px dashed #ccc',
                    borderRadius: 2,
                    p: 3,
                    textAlign: 'center',
                    cursor: 'pointer',
                    backgroundColor: isDragActive ? '#f0f8ff' : '#fafafa',
                    '&:hover': {
                        backgroundColor: '#f0f8ff',
                        borderColor: '#1976d2'
                    }
                }}
            >
                <input {...getInputProps()} />
                <CloudUpload sx={{ fontSize: 48, color: '#666', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                    {isDragActive ? 'Drop audio file here' : 'Drag & drop audio file here'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    or click to select file
                </Typography>
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                    Supported formats: MP3, WAV, M4A, FLAC, OGG
                </Typography>
            </Box>
            {/* Selected File Display */}
            {selectedFile && (
                <Box sx={{ mt: 2, p: 2, bgcolor: '#e3f2fd', borderRadius: 1 }}>
                    <Typography variant="body2" color="primary">
                        Selected: {selectedFile.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </Typography>
                </Box>
            )}
            {/* Status Bar (no progress bar) */}
            {(analysisStatus !== 'idle' || statusMessage) && (
                <Box sx={{ mt: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        {getStatusIcon()}
                        <Typography variant="body2" color="text.secondary">
                            {statusMessage}
                        </Typography>
                    </Box>
                </Box>
            )}
            {/* Analyze Button */}
            <Button
                variant="contained"
                color={analysisStatus === 'error' ? 'error' : 'primary'}
                fullWidth
                size="large"
                onClick={handleAnalyzeButton}
                disabled={isButtonDisabled()}
                startIcon={<Mic />}
                sx={{ mt: 2 }}
            >
                {getButtonText()}
            </Button>
        </Box>
    );
};

export default AudioAnalysis; 