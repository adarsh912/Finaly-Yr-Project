#!/usr/bin/env python3
"""
Single Model Predictor
Custom trained BERT-based prediction system for legal case analysis
"""

import os
import json
import torch
import numpy as np
import re
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional imports for additional functionality
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class SingleModelPredictor:
    """Single model predictor using custom trained BERT model"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the single model predictor
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.model_path = config.get('model_path', 'models/trained_model')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = config.get('max_length', 512)
        self.threshold = config.get('threshold', 0.25)
        self.max_predictions = config.get('max_predictions', 5)
        
        # Load model components
        self._load_model()
        
        print(f"🤖 Single Model Predictor Initialized")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Threshold: {self.threshold}")
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        
        try:
            # Load classes
            classes_path = os.path.join(self.model_path, 'classes.json')
            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    self.classes = json.load(f)
            else:
                print(f"⚠️  Classes file not found at {classes_path}")
                self.classes = []
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=len(self.classes),
                problem_type="multi_label_classification"
            )
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully with {len(self.classes)} classes")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Predict IPC sections from text input
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Prediction results
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Clean and prepare text
        text = self.clean_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        
        # Convert to numpy
        probabilities = probabilities.cpu().numpy().flatten()
        
        # Get top predictions above threshold
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > self.threshold:
                predictions.append({
                    'section': self.classes[i],
                    'confidence': float(prob)
                })
        
        # Sort by confidence and limit to max_predictions
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        predictions = predictions[:self.max_predictions]
        
        # Calculate overall confidence
        overall_confidence = np.mean([pred['confidence'] for pred in predictions]) if predictions else 0.0
        
        # Calculate processing time
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            processing_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            processing_time = 0.0
        
        return {
            'text': text[:200] + "..." if len(text) > 200 else text,
            'predictions': predictions,
            'confidence': overall_confidence,
            'total_sections_checked': len(self.classes),
            'processing_time': processing_time,
            'model_type': 'Custom Trained BERT Model'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text input"""
        
        if isinstance(text, list):
            text = ' '.join(text)
        elif not isinstance(text, str):
            text = str(text)
        
        # Basic cleaning
        text = ' '.join(text.split())  # Remove extra whitespace
        text = re.sub(r'Section\s+(\d+)', r'Section\1', text)  # Normalize section references
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers
        
        return text
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        if not SPEECH_AVAILABLE:
            raise ImportError("speech_recognition not available. Install with: pip install SpeechRecognition")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Import librosa for audio processing
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            raise ImportError("librosa and soundfile not available. Install with: pip install librosa soundfile")
        
        recognizer = sr.Recognizer()
        
        try:
            # Load audio with librosa and convert to WAV format
            y, sr_audio = librosa.load(audio_path, sr=16000)
            
            # Save as temporary WAV file
            temp_wav = "temp_audio.wav"
            sf.write(temp_wav, y, sr_audio)
            
            # Transcribe the WAV file
            with sr.AudioFile(temp_wav) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            
            # Clean up temporary file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
            return text
                    
        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            raise Exception(f"Failed to transcribe audio: {e}")
    
    def predict_audio(self, audio_path: str, top_k: int = None) -> Dict[str, Any]:
        """
        Predict IPC sections from audio file
        
        Args:
            audio_path (str): Path to audio file
            top_k (int): Number of top predictions to return
            
        Returns:
            Dict: Prediction results with same structure as predict_text
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Transcribe audio to text
        text = self.transcribe_audio(audio_path)
        
        if not text:
            return {
                'text': '',
                'predictions': [],
                'confidence': 0.0,
                'total_sections_checked': len(self.classes),
                'processing_time': 0.0,
                'model_type': 'Custom Trained BERT Model'
            }
        
        # Predict from transcribed text
        result = self.predict_text(text)
        predictions = result.get('predictions', [])
        
        # Limit to top_k if specified
        if top_k is not None:
            predictions = predictions[:top_k]
        
        # Calculate processing time
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            processing_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            processing_time = 0.0
        
        return {
            'text': text[:200] + "..." if len(text) > 200 else text,
            'predictions': predictions,
            'confidence': result.get('confidence', 0.0),
            'total_sections_checked': len(self.classes),
            'processing_time': processing_time,
            'model_type': 'Custom Trained BERT Model'
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Extracted text
        """
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 not available. Install with: pip install PyPDF2")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            str: Extracted text
        """
        if not OCR_AVAILABLE:
            raise ImportError("pytesseract not available. Install with: pip install pytesseract pillow")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to extract text from image: {e}")
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict IPC sections for multiple texts
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.predict_text(text)
                result['input_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'input_index': i
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        
        return {
            'model_type': 'Custom Trained BERT Model',
            'model_path': self.model_path,
            'device': str(self.device),
            'max_length': self.max_length,
            'threshold': self.threshold,
            'max_predictions': self.max_predictions,
            'num_classes': len(self.classes),
            'classes': self.classes[:10] + ['...'] if len(self.classes) > 10 else self.classes
        }
    
    def update_threshold(self, new_threshold: float):
        """Update prediction threshold"""
        
        if 0.0 <= new_threshold <= 1.0:
            self.threshold = new_threshold
            print(f"✅ Threshold updated to {new_threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def update_max_predictions(self, new_max: int):
        """Update maximum number of predictions"""
        
        if new_max > 0:
            self.max_predictions = new_max
            print(f"✅ Max predictions updated to {new_max}")
        else:
            raise ValueError("Max predictions must be greater than 0")

    def predict_text_with_transcription(self, text: str, transcription: str) -> Dict[str, Any]:
        """
        Predict IPC sections from text input with transcription
        
        Args:
            text (str): Input text
            transcription (str): Transcribed text
            
        Returns:
            Dict: Prediction results with transcription
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Clean and prepare text
        text = self.clean_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        
        # Convert to numpy
        probabilities = probabilities.cpu().numpy().flatten()
        
        # Get top predictions above threshold
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > self.threshold:
                predictions.append({
                    'section': self.classes[i],
                    'confidence': float(prob)
                })
        
        # Sort by confidence and limit to max_predictions
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        predictions = predictions[:self.max_predictions]
        
        # Calculate overall confidence
        overall_confidence = np.mean([pred['confidence'] for pred in predictions]) if predictions else 0.0
        
        # Calculate processing time
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            processing_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            processing_time = 0.0
        
        return {
            'predictions': predictions,
            'confidence': overall_confidence,
            'model_type': 'Custom Trained BERT Model',
            'transcription': transcription
        } 