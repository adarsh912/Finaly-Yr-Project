#!/usr/bin/env python3
"""
Unified Legal AI System
Integrates single model and multimodal prediction capabilities
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
import pandas as pd

from .single_model_predictor import SingleModelPredictor
from .multimodal_predictor import MultimodalPredictor


class UnifiedLegalAI:
    """Unified Legal AI System that combines single model and multimodal capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the unified legal AI system
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.single_predictor = SingleModelPredictor(config.get('single_model', {}))
        
        # Initialize multimodal predictor with correct parameters
        multimodal_config = config.get('multimodal', {})
        self.multimodal_predictor = MultimodalPredictor(
            model_path=multimodal_config.get('model_path', 'models/trained_model'),
            ipc_dataset_path=multimodal_config.get('ipc_dataset_path', 'data/ipc_sections.csv')
        )
        
        # Unified configuration
        self.unified_config = config.get('unified', {})
        self.default_mode = self.unified_config.get('default_mode', 'multimodal')
        self.fallback_to_single = self.unified_config.get('fallback_to_single', True)
        self.confidence_threshold = self.unified_config.get('confidence_threshold', 0.3)
        
        print("🚀 Unified Legal AI System Initialized")
        print("=" * 60)
        print(f"Default Mode: {self.default_mode}")
        print(f"Fallback to Single: {self.fallback_to_single}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print("=" * 60)
    
    def process_unified(self, input_data: str, input_type: str = 'text') -> Dict[str, Any]:
        """
        Process input using unified approach
        
        Args:
            input_data (str): Input text or file path
            input_type (str): Type of input ('text', 'audio', 'pdf', 'image')
            
        Returns:
            Dict: Unified analysis results
        """
        start_time = time.time()
        
        print(f"🔄 Processing {input_type.upper()} with Unified System")
        print("=" * 50)
        
        # Try multimodal first (default)
        if self.default_mode == 'multimodal':
            try:
                multimodal_result = self._process_multimodal(input_data, input_type)
                
                # Check if multimodal result is satisfactory
                if self._is_result_satisfactory(multimodal_result):
                    multimodal_result['processing_time'] = time.time() - start_time
                    multimodal_result['mode_used'] = 'multimodal'
                    return multimodal_result
                
            except Exception as e:
                print(f"⚠️  Multimodal processing failed: {e}")
        
        # Fallback to single model if needed
        if self.fallback_to_single:
            try:
                single_result = self._process_single(input_data, input_type)
                single_result['processing_time'] = time.time() - start_time
                single_result['mode_used'] = 'single'
                return single_result
                
            except Exception as e:
                print(f"⚠️  Single model processing failed: {e}")
        
        # If both fail, return error
        return {
            'error': 'Both multimodal and single model processing failed',
            'processing_time': time.time() - start_time,
            'mode_used': 'failed'
        }
    
    def _process_multimodal(self, input_data: str, input_type: str) -> Dict[str, Any]:
        """Process input using multimodal system"""
        
        if input_type == 'text':
            return self.multimodal_predictor.analyze_text(input_data)
        elif input_type == 'audio':
            return self.multimodal_predictor.analyze_audio(input_data)
        elif input_type == 'pdf':
            return self.multimodal_predictor.analyze_pdf(input_data)
        elif input_type == 'image':
            return self.multimodal_predictor.analyze_image(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    
    def _process_single(self, input_data: str, input_type: str) -> Dict[str, Any]:
        """Process input using single model system"""
        
        # Convert non-text inputs to text for single model
        if input_type == 'text':
            text = input_data
        elif input_type == 'audio':
            text = self.single_predictor.transcribe_audio(input_data)
        elif input_type == 'pdf':
            text = self.single_predictor.extract_text_from_pdf(input_data)
        elif input_type == 'image':
            text = self.single_predictor.extract_text_from_image(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        
        return self.single_predictor.predict_text(text)
    
    def _is_result_satisfactory(self, result: Dict[str, Any]) -> bool:
        """Check if multimodal result is satisfactory"""
        
        # Check if we have predictions
        if 'predictions' not in result or not result['predictions']:
            return False
        
        # Check confidence threshold
        if 'confidence' in result:
            return result['confidence'] >= self.confidence_threshold
        
        # Check individual prediction confidences
        if 'predictions' in result:
            max_confidence = max([pred.get('confidence', 0) for pred in result['predictions']])
            return max_confidence >= self.confidence_threshold
        
        return True
    
    def compare_modes(self, input_data: str, input_type: str = 'text') -> Dict[str, Any]:
        """
        Compare results from both single and multimodal modes
        
        Args:
            input_data (str): Input text or file path
            input_type (str): Type of input
            
        Returns:
            Dict: Comparison results
        """
        print("🔍 Comparing Single vs Multimodal Modes")
        print("=" * 50)
        
        start_time = time.time()
        
        # Get single model result
        single_result = None
        try:
            single_result = self._process_single(input_data, input_type)
        except Exception as e:
            print(f"⚠️  Single model failed: {e}")
        
        # Get multimodal result
        multimodal_result = None
        try:
            multimodal_result = self._process_multimodal(input_data, input_type)
        except Exception as e:
            print(f"⚠️  Multimodal failed: {e}")
        
        comparison = {
            'input_type': input_type,
            'single_model': single_result,
            'multimodal': multimodal_result,
            'comparison_time': time.time() - start_time
        }
        
        # Add comparison metrics
        if single_result and multimodal_result:
            comparison['metrics'] = self._compare_metrics(single_result, multimodal_result)
        
        return comparison
    
    def _compare_metrics(self, single_result: Dict, multimodal_result: Dict) -> Dict[str, Any]:
        """Compare metrics between single and multimodal results"""
        
        metrics = {}
        
        # Compare number of predictions
        single_count = len(single_result.get('predictions', []))
        multimodal_count = len(multimodal_result.get('predictions', []))
        metrics['prediction_count'] = {
            'single': single_count,
            'multimodal': multimodal_count,
            'difference': multimodal_count - single_count
        }
        
        # Compare confidence scores
        if 'confidence' in single_result and 'confidence' in multimodal_result:
            metrics['confidence'] = {
                'single': single_result['confidence'],
                'multimodal': multimodal_result['confidence'],
                'difference': multimodal_result['confidence'] - single_result['confidence']
            }
        
        # Compare processing time
        if 'processing_time' in single_result and 'processing_time' in multimodal_result:
            metrics['processing_time'] = {
                'single': single_result['processing_time'],
                'multimodal': multimodal_result['processing_time'],
                'ratio': multimodal_result['processing_time'] / single_result['processing_time']
            }
        
        return metrics
    
    def batch_process(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple inputs in batch
        
        Args:
            inputs (List[Dict]): List of input dictionaries with 'data' and 'type' keys
            
        Returns:
            List[Dict]: Batch processing results
        """
        print(f"📦 Batch Processing {len(inputs)} inputs")
        print("=" * 50)
        
        results = []
        
        for i, input_item in enumerate(inputs, 1):
            print(f"Processing item {i}/{len(inputs)}")
            
            try:
                result = self.process_unified(
                    input_item['data'], 
                    input_item.get('type', 'text')
                )
                result['input_index'] = i
                results.append(result)
                
            except Exception as e:
                print(f"⚠️  Failed to process item {i}: {e}")
                results.append({
                    'error': str(e),
                    'input_index': i,
                    'mode_used': 'failed'
                })
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and capabilities"""
        
        return {
            'model_type': 'Custom Trained BERT Model',
            'model_path': self.single_predictor.model_path,
            'device': str(self.single_predictor.device),
            'max_length': self.single_predictor.max_length,
            'threshold': self.single_predictor.threshold,
            'max_predictions': self.single_predictor.max_predictions,
            'total_sections': len(self.single_predictor.classes),
            'models': ['Custom Trained BERT Model', 'BERT', 'RoBERTa', 'DistilBERT', 'OpenAI GPT'],
            'multimodal_capabilities': ['text', 'audio'],
            'asr_models': ['Whisper', 'Faster-Whisper', 'Google Speech'],
            'offline_capable': True,
            'version': '2.1.0'
        } 