export interface IPCClassification {
    section: string;
    confidence: number;
}

export interface ModelPerformance {
    model_name: string;
    predictions: IPCClassification[];
    confidence_score: number;
    processing_time: number;
}

export interface AnalysisResult {
    input_type: string;
    input_content: string;
    analysis_mode: string;
    timestamp: string;
    results: {
        best_model: string;
        best_predictions: IPCClassification[];
        all_model_results: ModelPerformance[];
        total_processing_time: number;
    };
    metadata?: {
        audio_duration?: number;
        transcription_models?: string[];
        transcription_confidence?: number;
    };
}

export interface APIResponse {
    success: boolean;
    data?: AnalysisResult;
    error?: string;
}

export interface AnalysisRequest {
    input_type: 'text' | 'audio';
    content: string | File;
    mode: 'single' | 'multimodal';
} 