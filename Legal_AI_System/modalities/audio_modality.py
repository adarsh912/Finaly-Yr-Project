#!/usr/bin/env python3
"""
Audio Modality Analyzer
Uses the custom trained BERT model for audio-based legal document analysis
"""

import time
import os
from typing import List, Dict, Any
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import openai
from tqdm import tqdm
import pandas as pd
import librosa
import speech_recognition as sr
from pydub import AudioSegment

from core.model_performance import ModelPerformance

class AudioModalityAnalyzer:
    """Audio modality analyzer using the custom trained BERT model"""
    
    def __init__(self, ipc_dataset_path: str):
        """
        Initialize audio modality analyzer
        
        Args:
            ipc_dataset_path (str): Path to IPC dataset
        """
        self.ipc_dataset_path = ipc_dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load IPC dataset
        self.load_ipc_dataset()
        
        # Initialize models
        self.models = {}
        self.initialize_models()
        
        print(f"🎵 Audio Modality Analyzer initialized with {len(self.models)} models")
    
    def load_ipc_dataset(self):
        """Load IPC dataset"""
        try:
            self.df = pd.read_csv(self.ipc_dataset_path)
            self.offense_texts = self.df['Offense'].fillna("").tolist()
            print(f"✅ Loaded {len(self.offense_texts)} IPC offenses for audio analysis")
        except Exception as e:
            print(f"⚠️  Error loading IPC dataset: {e}")
            self.df = None
            self.offense_texts = []
    
    def initialize_models(self):
        """Initialize the custom trained BERT model for audio analysis"""
        print("🤖 Initializing Audio Models...")
        
        # 1. Custom Trained BERT Model (Primary Model)
        try:
            self.models['Custom Trained BERT'] = self._init_trained_bert_model()
            print("✅ Custom Trained BERT Model initialized")
        except Exception as e:
            print(f"⚠️  Custom Trained BERT Model initialization failed: {e}")
    
    def _init_trained_bert_model(self):
        """Initialize the custom trained BERT model from models/trained_model"""
        # Get the path to the trained model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'models', 'trained_model')
        
        if not os.path.exists(model_path):
            raise Exception(f"Trained model not found at: {model_path}")
        
        print(f"Loading trained model from: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model = model.to(self.device)
        model.eval()
        
        return {
            'type': 'transformer',
            'tokenizer': tokenizer,
            'model': model,
            'name': 'Custom-Trained-BERT',
            'path': model_path
        }
    
    def extract_text_from_audio(self, audio_path: str) -> str:
        """
        Extract text from audio file using speech recognition
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            str: Extracted text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Convert audio to WAV format if needed
            audio = self._load_audio(audio_path)
            
            # Extract text using speech recognition
            text = self._speech_to_text(audio)
            
            # Clean the extracted text
            text = self._preprocess_audio_text(text)
            
            return text
            
        except Exception as e:
            raise Exception(f"Failed to extract text from audio: {e}")
    
    def _load_audio(self, audio_path: str) -> AudioSegment:
        """
        Load audio file and convert to WAV format
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            AudioSegment: Loaded audio
        """
        # Import librosa for audio processing
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            raise ImportError("librosa and soundfile not available. Install with: pip install librosa soundfile")
        
        # Load audio with librosa and convert to WAV format
        y, sr_audio = librosa.load(audio_path, sr=16000)
        
        # Save as temporary WAV file
        temp_wav = "temp_audio.wav"
        sf.write(temp_wav, y, sr_audio)
        
        # Return the path to the temporary WAV file
        return temp_wav
    
    def _speech_to_text(self, audio_path: str) -> str:
        """
        Convert speech to text using speech recognition
        
        Args:
            audio_path (str): Path to audio file (WAV format)
            
        Returns:
            str: Transcribed text
        """
        try:
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                # Read audio data
                audio_data = recognizer.record(source)
                
                # Recognize speech
                text = recognizer.recognize_google(audio_data)
                
                return text
                
        except sr.UnknownValueError:
            print("⚠️  Speech recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"⚠️  Speech recognition service error: {e}")
            return ""
        finally:
            # Clean up temporary file if it was created by _load_audio
            if audio_path == "temp_audio.wav" and os.path.exists(audio_path):
                os.remove(audio_path)
    
    def _preprocess_audio_text(self, text: str) -> str:
        """
        Preprocess extracted audio text
        
        Args:
            text (str): Raw transcribed text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common speech recognition artifacts
        text = text.replace('um', '')
        text = text.replace('uh', '')
        text = text.replace('ah', '')
        
        # Remove special characters but keep legal terms
        import re
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        return text.strip()
    
    def get_embedding_transformer(self, text: str, model_info: Dict) -> np.ndarray:
        """Get embedding using transformer model"""
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    
    def get_embedding(self, text: str, model_info: Dict) -> np.ndarray:
        """Get embedding based on model type"""
        if model_info['type'] == 'transformer':
            return self.get_embedding_transformer(text, model_info)
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")
    
    def compute_embeddings_for_model(self, model_name: str) -> List[np.ndarray]:
        """Compute embeddings for all IPC offenses using specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.models[model_name]
        embeddings = []
        
        print(f"Computing embeddings for {len(self.offense_texts)} offenses using {model_name}...")
        
        for text in tqdm(self.offense_texts, desc=f"Computing {model_name} embeddings"):
            try:
                embedding = self.get_embedding(text, model_info)
                embeddings.append(embedding)
            except Exception as e:
                print(f"⚠️  Error computing embedding for text: {e}")
                # Use zero vector as fallback
                embedding_dim = 768  # Default BERT embedding dimension
                embeddings.append(np.zeros(embedding_dim))
        
        return embeddings
    
    def predict_with_model(self, audio_path: str, model_name: str, top_n: int = 5) -> List[Dict]:
        """
        Predict IPC sections from audio using a specific model
        
        Args:
            audio_path (str): Path to audio file
            model_name (str): Name of the model to use
            top_n (int): Number of top predictions to return
            
        Returns:
            List[Dict]: List of predictions with confidence scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Extract text from audio
        text = self.extract_text_from_audio(audio_path)
        
        model_info = self.models[model_name]
        
        # Get embedding for extracted text
        text_embedding = self.get_embedding(text, model_info)
        
        # Compute embeddings for all IPC offenses if not already computed
        if not hasattr(self, f'{model_name}_embeddings'):
            setattr(self, f'{model_name}_embeddings', self.compute_embeddings_for_model(model_name))
        
        offense_embeddings = getattr(self, f'{model_name}_embeddings')
        
        # Compute similarities
        similarities = cosine_similarity([text_embedding], offense_embeddings)[0]
        
        # Get top N predictions
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                'section': self.df.iloc[idx]['Section'] if self.df is not None else f"Section_{idx}",
                'confidence': float(similarities[idx]),
                'description': self.offense_texts[idx] if idx < len(self.offense_texts) else ""
            })
        
        return predictions
    
    def analyze_with_model(self, audio_path: str, model_name: str, true_sections: List[str] = None) -> ModelPerformance:
        """
        Analyze audio with a specific model and return performance metrics
        
        Args:
            audio_path (str): Path to audio file
            model_name (str): Name of the model to use
            true_sections (List[str]): Ground truth sections for evaluation
            
        Returns:
            ModelPerformance: Performance metrics and predictions
        """
        start_time = time.time()
        
        # Get predictions
        predictions = self.predict_with_model(audio_path, model_name, top_n=10)
        predicted_sections = [pred['section'] for pred in predictions]
        confidence_scores = [pred['confidence'] for pred in predictions]
        
        # Calculate metrics
        accuracy, precision, recall, f1_score = self.calculate_metrics(true_sections, predicted_sections)
        
        # Calculate average confidence
        confidence_score = np.mean(confidence_scores) if confidence_scores else 0.0
        
        inference_time = time.time() - start_time
        
        return ModelPerformance(
            model_name=model_name,
            modality="audio",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            inference_time=inference_time,
            confidence_score=confidence_score,
            predicted_sections=predicted_sections,
            true_sections=true_sections or []
        )
    
    def calculate_metrics(self, true_sections: List[str], predicted_sections: List[str]) -> tuple:
        """
        Calculate performance metrics
        
        Args:
            true_sections (List[str]): Ground truth sections
            predicted_sections (List[str]): Predicted sections
            
        Returns:
            tuple: (accuracy, precision, recall, f1_score)
        """
        if not true_sections or not predicted_sections:
            return 0.0, 0.0, 0.0, 0.0
        
        # Convert to sets for easier comparison
        true_set = set(true_sections)
        pred_set = set(predicted_sections)
        
        # Calculate metrics
        if len(pred_set) == 0:
            precision = 0.0
        else:
            precision = len(true_set.intersection(pred_set)) / len(pred_set)
        
        if len(true_set) == 0:
            recall = 0.0
        else:
            recall = len(true_set.intersection(pred_set)) / len(true_set)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Accuracy (exact match)
        accuracy = 1.0 if true_set == pred_set else 0.0
        
        return accuracy, precision, recall, f1_score
    
    def analyze_with_all_models(self, audio_path: str, true_sections: List[str] = None) -> List[ModelPerformance]:
        """
        Analyze audio with all available models
        
        Args:
            audio_path (str): Path to audio file
            true_sections (List[str]): Ground truth sections for evaluation
        
        Returns:
            List[ModelPerformance]: Performance metrics for all models
        """
        results = []
        for model_name in self.models.keys():
            try:
                result = self.analyze_with_model(audio_path, model_name, true_sections)
                if isinstance(result, ModelPerformance):
                    results.append(result)
            except Exception as e:
                print(f"⚠️  Error analyzing with {model_name}: {e}")
        return results
    
    def get_model_comparison_table(self, results: List[ModelPerformance]) -> pd.DataFrame:
        """
        Create a comparison table for model results
        
        Args:
            results (List[ModelPerformance]): Model performance results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        data = []
        for result in results:
            data.append({
                'Model': result.model_name,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1_Score': result.f1_score,
                'Inference_Time': result.inference_time,
                'Num_Predictions': len(result.predicted_sections)
            })
        
        return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    # Initialize audio analyzer
    analyzer = AudioModalityAnalyzer('ipc_sections.csv')
    
    # Test case (you would need a real audio file)
    test_audio = "sample_legal_audio.wav"  # Replace with actual audio path
    true_sections = ["IPC_380", "IPC_457"]
    
    # Check if test audio exists
    if os.path.exists(test_audio):
        # Analyze with all models
        results = analyzer.analyze_with_all_models(test_audio, true_sections)
        
        # Print comparison table
        comparison_table = analyzer.get_model_comparison_table(results)
        print("\n📊 Model Comparison Table:")
        print(comparison_table.to_string(index=False))
    else:
        print(f"⚠️  Test audio file not found: {test_audio}")
        print("Please provide a valid audio file for testing.") 