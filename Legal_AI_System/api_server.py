from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime
import tempfile
import shutil

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.single_model_predictor import SingleModelPredictor
from core.multimodal_predictor import MultimodalPredictor
from core.unified_legal_ai import UnifiedLegalAI

# Load config from JSON file
with open('config/default_config.json', 'r') as f:
    DEFAULT_CONFIG = json.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize predictors
single_predictor = None
multimodal_predictor = None
unified_ai = None

def initialize_models():
    """Initialize the AI models"""
    global single_predictor, multimodal_predictor, unified_ai
    
    try:
        print("Initializing Legal AI System models...")
        
        # Initialize single model predictor
        single_predictor = SingleModelPredictor(DEFAULT_CONFIG)
        
        # Initialize multimodal predictor with correct parameters
        multimodal_predictor = MultimodalPredictor(
            model_path=DEFAULT_CONFIG['single_model']['model_path'],
            ipc_dataset_path=DEFAULT_CONFIG['multimodal']['ipc_dataset_path']
        )
        
        # Create a unified config that matches what UnifiedLegalAI expects
        unified_config = {
            'single_model': DEFAULT_CONFIG['single_model'],
            'multimodal': {
                'model_path': DEFAULT_CONFIG['single_model']['model_path'],
                'ipc_dataset_path': DEFAULT_CONFIG['multimodal']['ipc_dataset_path']
            },
            'unified': DEFAULT_CONFIG['unified']
        }
        
        # Initialize unified legal AI
        unified_ai = UnifiedLegalAI(unified_config)
        
        print("Models initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'success': True,
        'status': 'running',
        'models_loaded': single_predictor is not None and multimodal_predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify({
        'success': True,
        'models': {
            'single_model': 'Trained InLegalBERT',
            'multimodal': 'Multimodal Ensemble',
            'asr_models': ['Whisper', 'Faster-Whisper', 'Google Speech Recognition']
        }
    })

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze text input"""
    try:
        data = request.get_json()
        text = data.get('content', '').strip()
        mode = data.get('mode', 'single')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text content provided'
            }), 400
        
        print(f"Analyzing text in {mode} mode: {text[:100]}...")
        
        # Perform analysis based on mode
        if mode == 'single':
            if single_predictor is None:
                return jsonify({
                    'success': False,
                    'error': 'Single model predictor not initialized'
                }), 500
            
            result = single_predictor.predict_text(text)
            
            # Format result for frontend
            analysis_result = {
                'input_type': 'text',
                'input_content': text,
                'analysis_mode': 'single',
                'timestamp': datetime.now().isoformat(),
                'results': {
                    'best_model': 'Trained InLegalBERT',
                    'best_predictions': result['predictions'],
                    'all_model_results': [{
                        'model_name': 'Trained InLegalBERT',
                        'predictions': result['predictions'],
                        'confidence_score': result['confidence'],
                        'processing_time': result['processing_time']
                    }],
                    'total_processing_time': result['processing_time']
                }
            }
            
        else:  # multimodal
            if multimodal_predictor is None:
                return jsonify({
                    'success': False,
                    'error': 'Multimodal predictor not initialized'
                }), 500
            
            result = multimodal_predictor.analyze_text(text)
            
            # Format result for frontend
            analysis_result = {
                'input_type': 'text',
                'input_content': text,
                'analysis_mode': 'multimodal',
                'timestamp': datetime.now().isoformat(),
                'results': {
                    'best_model': result.get('model_type', 'Multimodal'),
                    'best_predictions': result.get('predictions', []),
                    'all_model_results': [{
                        'model_name': result.get('model_type', 'Multimodal'),
                        'predictions': result.get('predictions', []),
                        'confidence_score': result.get('confidence', 0.0),
                        'processing_time': result.get('processing_time', 0.0)
                    }],
                    'total_processing_time': result.get('processing_time', 0.0)
                }
            }
        
        return jsonify({
            'success': True,
            'data': analysis_result
        })
        
    except Exception as e:
        print(f"Error in text analysis: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/analyze/audio', methods=['POST'])
def analyze_audio():
    """Analyze audio input"""
    try:
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio']
        mode = request.form.get('mode', 'single')
        
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No audio file selected'
            }), 400
        
        # Validate file type
        allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': f'Unsupported file type. Allowed: {", ".join(allowed_extensions)}'
            }), 400
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, audio_file.filename)
        audio_file.save(temp_path)
        
        try:
            print(f"Analyzing audio file: {audio_file.filename} in {mode} mode")
            
            # Perform analysis based on mode
            if mode == 'single':
                if single_predictor is None:
                    return jsonify({
                        'success': False,
                        'error': 'Single model predictor not initialized'
                    }), 500
                
                result = single_predictor.predict_audio(temp_path)
                
                # Format result for frontend
                analysis_result = {
                    'input_type': 'audio',
                    'input_content': f"Audio file: {audio_file.filename}",
                    'analysis_mode': 'single',
                    'timestamp': datetime.now().isoformat(),
                    'results': {
                        'best_model': 'Trained InLegalBERT',
                        'best_predictions': result['predictions'],
                        'all_model_results': [{
                            'model_name': 'Trained InLegalBERT',
                            'predictions': result['predictions'],
                            'confidence_score': result['confidence'],
                            'processing_time': result['processing_time']
                        }],
                        'total_processing_time': result['processing_time']
                    },
                    'metadata': {
                        'audio_duration': result.get('audio_duration', None),
                        'transcription_models': result.get('transcription_models', None),
                        'transcription_confidence': result.get('transcription_confidence', None)
                    }
                }
                
            else:  # multimodal
                if multimodal_predictor is None:
                    return jsonify({
                        'success': False,
                        'error': 'Multimodal predictor not initialized'
                    }), 500
                
                result = multimodal_predictor.analyze_audio(temp_path)
                
                # Format result for frontend
                analysis_result = {
                    'input_type': 'audio',
                    'input_content': f"Audio file: {audio_file.filename}",
                    'analysis_mode': 'multimodal',
                    'timestamp': datetime.now().isoformat(),
                    'results': {
                        'best_model': result.get('model_type', 'Multimodal'),
                        'best_predictions': result.get('predictions', []),
                        'all_model_results': [{
                            'model_name': result.get('model_type', 'Multimodal'),
                            'predictions': result.get('predictions', []),
                            'confidence_score': result.get('confidence', 0.0),
                            'processing_time': result.get('processing_time', 0.0)
                        }],
                        'total_processing_time': result.get('processing_time', 0.0)
                    },
                    'metadata': {
                        'audio_duration': result.get('audio_duration', None),
                        'transcription_models': result.get('transcription_models', None),
                        'transcription_confidence': result.get('transcription_confidence', None)
                    }
                }
            
            return jsonify({
                'success': True,
                'data': analysis_result
            })
            
        finally:
            # Clean up temporary file
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"Error in audio analysis: {e}")
        return jsonify({
            'success': False,
            'error': f'Audio analysis failed: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'Legal AI System API is running',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Initialize models before starting server
    if initialize_models():
        print("Starting Legal AI System API server...")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("Failed to initialize models. Exiting.")
        sys.exit(1) 