const API_BASE_URL = 'http://localhost:5001/api';

export interface AnalysisResult {
  input_type: string;
  input_content: string;
  analysis_mode: string;
  timestamp: string;
  results: {
    best_model: string;
    best_predictions: Array<{
      section: string;
      confidence: number;
    }>;
    all_model_results: Array<{
      model_name: string;
      predictions: Array<{
        section: string;
        confidence: number;
      }>;
      confidence_score: number;
      processing_time: number;
    }>;
    total_processing_time: number;
  };
  metadata?: {
    audio_duration?: number;
    transcription_models?: string[];
    transcription_confidence?: number;
  };
}

class ApiService {
  private addDebugLog: ((message: string) => void) | null = null;

  setDebugLogger(logger: (message: string) => void) {
    this.addDebugLog = logger;
  }

  private log(message: string) {
    if (this.addDebugLog) {
      this.addDebugLog(message);
    }
    console.log(`[API] ${message}`);
  }

  async analyzeText(text: string, mode: 'single' | 'multimodal'): Promise<AnalysisResult> {
    this.log(`Making text analysis request: mode=${mode}, text_length=${text.length}`);
    
    try {
      const response = await fetch(`${API_BASE_URL}/analyze/text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: text,
          mode: mode
        }),
      });

      this.log(`Text analysis response status: ${response.status}`);

      if (!response.ok) {
        const errorData = await response.json();
        this.log(`Text analysis error: ${errorData.error || 'Unknown error'}`);
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const data = await response.json();
      this.log(`Text analysis successful: ${data.data?.results?.best_predictions?.length || 0} predictions`);
      return data.data;
    } catch (error) {
      this.log(`Text analysis request failed: ${error}`);
      throw error;
    }
  }

  async analyzeAudio(formData: FormData, onProgress?: (progress: number) => void): Promise<AnalysisResult> {
    this.log(`Making audio analysis request`);
    
    try {
      const xhr = new XMLHttpRequest();
      
      return new Promise((resolve, reject) => {
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable && onProgress) {
            const progress = (event.loaded / event.total) * 100;
            onProgress(progress);
            this.log(`Upload progress: ${progress.toFixed(1)}%`);
          }
        });

        xhr.addEventListener('load', () => {
          this.log(`Audio analysis response status: ${xhr.status}`);
          
          if (xhr.status === 200) {
            try {
              const data = JSON.parse(xhr.responseText);
              this.log(`Audio analysis successful: ${data.data?.results?.best_predictions?.length || 0} predictions`);
              resolve(data.data);
            } catch (error) {
              this.log(`Audio analysis response parsing failed: ${error}`);
              reject(new Error('Invalid response format'));
            }
          } else {
            try {
              const errorData = JSON.parse(xhr.responseText);
              this.log(`Audio analysis error: ${errorData.error || 'Unknown error'}`);
              reject(new Error(errorData.error || `HTTP ${xhr.status}`));
            } catch {
              this.log(`Audio analysis error: HTTP ${xhr.status}`);
              reject(new Error(`HTTP ${xhr.status}`));
            }
          }
        });

        xhr.addEventListener('error', () => {
          this.log(`Audio analysis network error`);
          reject(new Error('Network error'));
        });

        xhr.open('POST', `${API_BASE_URL}/analyze/audio`);
        xhr.send(formData);
      });
    } catch (error) {
      this.log(`Audio analysis request failed: ${error}`);
      throw error;
    }
  }

  async getStatus(): Promise<any> {
    this.log('Making status request');
    
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      this.log(`Status response status: ${response.status}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      this.log(`Status successful: ${data.status}`);
      return data;
    } catch (error) {
      this.log(`Status request failed: ${error}`);
      throw error;
    }
  }

  async getModels(): Promise<any> {
    this.log('Making models request');
    
    try {
      const response = await fetch(`${API_BASE_URL}/models`);
      this.log(`Models response status: ${response.status}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      this.log(`Models successful: ${data.models ? Object.keys(data.models).length : 0} models`);
      return data;
    } catch (error) {
      this.log(`Models request failed: ${error}`);
      throw error;
    }
  }
}

export const apiService = new ApiService();

export const handleApiError = (error: any): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
}; 