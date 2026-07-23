#!/usr/bin/env python3
"""
Multimodal Predictor
Simplified multimodal legal AI system
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Import modality analyzers (simplified versions)
from modalities.text_modality import TextModalityAnalyzer
from modalities.audio_modality import AudioModalityAnalyzer


class MultimodalPredictor:
    """Multimodal legal AI system"""
    
    def __init__(self, model_path: str, ipc_dataset_path: str):
        """Initialize the multimodal predictor"""
        self.text_analyzer = TextModalityAnalyzer(ipc_dataset_path)
        self.audio_analyzer = AudioModalityAnalyzer(ipc_dataset_path)
        
        print("🚀 Multimodal Predictor Initialized")
    
    def analyze_text(self, text: str, true_sections=None):
        """Analyze text input"""
        start_time = time.time()
        
        try:
            results = self.text_analyzer.analyze_with_all_models(text, true_sections)
            best_result = self._select_best_result(results)
            
            return {
                'modality': 'text',
                'input': text,
                'predictions': best_result.get('predictions', []),
                'confidence': best_result.get('confidence', 0.0),
                'processing_time': time.time() - start_time,
                'model_type': 'multimodal'
            }
        except Exception as e:
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def analyze_audio(self, audio_path: str, true_sections=None):
        """Analyze audio input"""
        start_time = time.time()
        
        try:
            results = self.audio_analyzer.analyze_with_all_models(audio_path, true_sections)
            best_result = self._select_best_result(results)
            
            return {
                'modality': 'audio',
                'input_path': audio_path,
                'predictions': best_result.get('predictions', []),
                'confidence': best_result.get('confidence', 0.0),
                'processing_time': time.time() - start_time,
                'model_type': 'multimodal'
            }
        except Exception as e:
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _select_best_result(self, results: List) -> Dict[str, Any]:
        """Select the best result based on confidence"""
        if not results:
            return {}
        
        # Filter out invalid results (e.g., strings, None)
        results = [r for r in results if hasattr(r, 'f1_score') or isinstance(r, dict)]
        if not results:
            return {}
        
        # Check if results are ModelPerformance objects or dictionaries
        if hasattr(results[0], 'f1_score'):
            # ModelPerformance objects - sort by f1_score
            sorted_results = sorted(results, key=lambda x: x.f1_score, reverse=True)
            best_result = sorted_results[0]
            
            # Convert ModelPerformance to dictionary format with proper section formatting
            predictions = []
            for section in best_result.predicted_sections:
                # Convert IPC_140 format to Section 140 format
                if section.startswith('IPC_'):
                    section_number = section.replace('IPC_', '')
                    formatted_section = f"Section {section_number}"
                else:
                    formatted_section = section
                
                predictions.append({
                    'section': formatted_section, 
                    'confidence': best_result.confidence_score
                })
            
            return {
                'predictions': predictions,
                'confidence': best_result.confidence_score,
                'f1_score': best_result.f1_score,
                'model_name': best_result.model_name
            }
        else:
            # Dictionary format - sort by confidence
            sorted_results = sorted(results, key=lambda x: x.get('confidence', 0), reverse=True)
            return sorted_results[0] 