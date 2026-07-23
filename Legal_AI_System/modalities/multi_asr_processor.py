#!/usr/bin/env python3
"""
Multi-ASR Processor
Supports multiple LLM-based and traditional ASR backends for audio-to-text conversion
"""

import os
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

# Import ASR backends
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import speech_recognition as sr
    GOOGLE_ASR_AVAILABLE = True
except ImportError:
    GOOGLE_ASR_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

@dataclass
class ASRResult:
    """Result from ASR processing"""
    text: str
    confidence: float
    processing_time: float
    model_name: str
    language: str = "en"
    word_timestamps: Optional[List[Dict]] = None

class MultiASRProcessor:
    """Multi-ASR processor supporting multiple backends"""
    
    def __init__(self, models_config: Dict[str, Any] = None):
        """
        Initialize multi-ASR processor
        
        Args:
            models_config: Configuration for different ASR models
        """
        self.models_config = models_config or self._get_default_config()
        self.models = {}
        self.initialize_models()
        
        print(f"🎤 Multi-ASR Processor initialized with {len(self.models)} models")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ASR models"""
        return {
            "whisper": {
                "model_size": "base",  # tiny, base, small, medium, large
                "device": "cpu",
                "language": "en"
            },
            "faster_whisper": {
                "model_size": "base",
                "device": "cpu",
                "compute_type": "int8",
                "language": "en"
            },
            "google_speech": {
                "language": "en-IN",
                "timeout": 30
            }
        }
    
    def initialize_models(self):
        """Initialize all available ASR models"""
        print("🤖 Initializing ASR Models...")
        
        # Initialize OpenAI Whisper
        if WHISPER_AVAILABLE and "whisper" in self.models_config:
            try:
                config = self.models_config["whisper"]
                model_size = config.get("model_size", "base")
                print(f"Loading Whisper model: {model_size}")
                
                # Try to load from local models folder first
                local_model_path = f"models/whisper_models/{model_size}.pt"
                if os.path.exists(local_model_path):
                    print(f"Loading from local path: {local_model_path}")
                    self.models["whisper"] = {
                        "type": "whisper",
                        "model": whisper.load_model(local_model_path),
                        "config": config,
                        "name": f"Whisper-{model_size.upper()}-Local"
                    }
                else:
                    # Fallback to cache
                    print(f"Local model not found, loading from cache: {model_size}")
                    self.models["whisper"] = {
                        "type": "whisper",
                        "model": whisper.load_model(model_size),
                        "config": config,
                        "name": f"Whisper-{model_size.upper()}"
                    }
                print("✅ Whisper model loaded")
            except Exception as e:
                print(f"⚠️  Whisper initialization failed: {e}")
        
        # Initialize Faster Whisper
        if FASTER_WHISPER_AVAILABLE and "faster_whisper" in self.models_config:
            try:
                config = self.models_config["faster_whisper"]
                model_size = config.get("model_size", "base")
                device = config.get("device", "cpu")
                compute_type = config.get("compute_type", "int8")
                
                print(f"Loading Faster Whisper model: {model_size}")
                
                # Faster Whisper loads from cache automatically
                self.models["faster_whisper"] = {
                    "type": "faster_whisper",
                    "model": WhisperModel(model_size, device=device, compute_type=compute_type),
                    "config": config,
                    "name": f"Faster-Whisper-{model_size.upper()}"
                }
                print("✅ Faster Whisper model loaded")
            except Exception as e:
                print(f"⚠️  Faster Whisper initialization failed: {e}")
        
        # Initialize Google Speech Recognition
        if GOOGLE_ASR_AVAILABLE and "google_speech" in self.models_config:
            try:
                config = self.models_config["google_speech"]
                self.models["google_speech"] = {
                    "type": "google_speech",
                    "config": config,
                    "name": "Google Speech Recognition"
                }
                print("✅ Google Speech Recognition initialized")
            except Exception as e:
                print(f"⚠️  Google Speech Recognition initialization failed: {e}")
    
    def transcribe_with_whisper(self, audio_path: str) -> ASRResult:
        """Transcribe audio using OpenAI Whisper"""
        model_info = self.models["whisper"]
        model = model_info["model"]
        config = model_info["config"]
        
        start_time = time.time()
        
        try:
            # Transcribe audio
            result = model.transcribe(
                audio_path,
                language=config.get("language", "en"),
                fp16=False  # Use CPU
            )
            
            processing_time = time.time() - start_time
            
            return ASRResult(
                text=result["text"].strip(),
                confidence=result.get("confidence", 0.0),
                processing_time=processing_time,
                model_name=model_info["name"],
                language=config.get("language", "en"),
                word_timestamps=result.get("segments", [])
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ASRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                model_name=model_info["name"],
                error=str(e)
            )
    
    def transcribe_with_faster_whisper(self, audio_path: str) -> ASRResult:
        """Transcribe audio using Faster Whisper"""
        model_info = self.models["faster_whisper"]
        model = model_info["model"]
        config = model_info["config"]
        
        start_time = time.time()
        
        try:
            # Transcribe audio
            segments, info = model.transcribe(
                audio_path,
                language=config.get("language", "en"),
                beam_size=5
            )
            
            # Collect text and confidence
            text_parts = []
            confidences = []
            
            for segment in segments:
                text_parts.append(segment.text)
                confidences.append(segment.avg_logprob)
            
            text = " ".join(text_parts).strip()
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            return ASRResult(
                text=text,
                confidence=avg_confidence,
                processing_time=processing_time,
                model_name=model_info["name"],
                language=config.get("language", "en")
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ASRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                model_name=model_info["name"],
                error=str(e)
            )
    
    def transcribe_with_google_speech(self, audio_path: str) -> ASRResult:
        """Transcribe audio using Google Speech Recognition"""
        model_info = self.models["google_speech"]
        config = model_info["config"]
        
        start_time = time.time()
        
        try:
            # Convert audio to WAV if needed using librosa
            if not LIBROSA_AVAILABLE:
                raise ImportError("librosa not available for audio conversion")
            
            # Load and convert audio
            y, sr_audio = librosa.load(audio_path, sr=16000)
            temp_wav = "temp_google_asr.wav"
            sf.write(temp_wav, y, sr_audio)
            
            # Transcribe
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_wav) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(
                    audio_data,
                    language=config.get("language", "en-IN")
                )
            
            # Clean up
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            
            processing_time = time.time() - start_time
            
            return ASRResult(
                text=text.strip(),
                confidence=0.8,  # Google doesn't provide confidence
                processing_time=processing_time,
                model_name=model_info["name"],
                language=config.get("language", "en-IN")
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            # Clean up on error
            if os.path.exists("temp_google_asr.wav"):
                os.remove("temp_google_asr.wav")
            
            return ASRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                model_name=model_info["name"],
                error=str(e)
            )
    
    def transcribe_audio(self, audio_path: str, model_name: str = None) -> ASRResult:
        """
        Transcribe audio using specified model or all models
        
        Args:
            audio_path: Path to audio file
            model_name: Specific model to use (if None, uses all)
            
        Returns:
            ASRResult or List[ASRResult]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if model_name:
            # Use specific model
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available")
            
            return self._transcribe_with_model(audio_path, model_name)
        else:
            # Use all available models
            results = []
            for name in self.models.keys():
                try:
                    result = self._transcribe_with_model(audio_path, name)
                    results.append(result)
                except Exception as e:
                    print(f"⚠️  Error with {name}: {e}")
            
            return results
    
    def _transcribe_with_model(self, audio_path: str, model_name: str) -> ASRResult:
        """Transcribe audio with specific model"""
        model_info = self.models[model_name]
        model_type = model_info["type"]
        
        if model_type == "whisper":
            return self.transcribe_with_whisper(audio_path)
        elif model_type == "faster_whisper":
            return self.transcribe_with_faster_whisper(audio_path)
        elif model_type == "google_speech":
            return self.transcribe_with_google_speech(audio_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def compare_results(self, results: List[ASRResult]) -> Dict[str, Any]:
        """
        Compare results from multiple ASR models
        
        Args:
            results: List of ASR results
            
        Returns:
            Comparison analysis
        """
        if not results:
            return {"error": "No results to compare"}
        
        # Basic comparison
        comparison = {
            "total_models": len(results),
            "models": [],
            "best_model": None,
            "text_variations": [],
            "processing_times": {}
        }
        
        # Analyze each result
        for result in results:
            model_analysis = {
                "name": result.model_name,
                "text": result.text,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "text_length": len(result.text),
                "has_error": hasattr(result, 'error')
            }
            
            comparison["models"].append(model_analysis)
            comparison["processing_times"][result.model_name] = result.processing_time
        
        # Find best model (highest confidence)
        valid_results = [r for r in results if r.text and not hasattr(r, 'error')]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x.confidence)
            comparison["best_model"] = best_result.model_name
            comparison["best_text"] = best_result.text
            comparison["best_confidence"] = best_result.confidence
        
        # Analyze text variations
        texts = [r.text for r in results if r.text]
        comparison["text_variations"] = list(set(texts))
        comparison["unique_texts"] = len(set(texts))
        
        return comparison
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models"""
        info = {}
        for name, model_info in self.models.items():
            info[name] = {
                "name": model_info["name"],
                "type": model_info["type"],
                "config": model_info["config"]
            }
        return info 