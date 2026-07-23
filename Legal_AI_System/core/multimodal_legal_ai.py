#!/usr/bin/env python3
"""
Multimodal Legal AI System
Main orchestrator for multimodal legal case analysis with multiple LLM/Transformer models
"""

import os
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Import modality-specific modules
from modalities.text_modality import TextModalityAnalyzer
from modalities.audio_modality import AudioModalityAnalyzer
from core.evaluation_metrics import EvaluationMetrics
from core.model_performance import ModelPerformance

class MultimodalLegalAI:
    """Main multimodal legal AI system"""
    
    def __init__(self, ipc_dataset_path='ipc_sections.csv'):
        """
        Initialize the multimodal legal AI system
        
        Args:
            ipc_dataset_path (str): Path to IPC dataset
        """
        self.ipc_dataset_path = ipc_dataset_path
        self.evaluation_metrics = EvaluationMetrics()
        
        # Initialize modality analyzers
        self.text_analyzer = TextModalityAnalyzer(ipc_dataset_path)
        self.audio_analyzer = AudioModalityAnalyzer(ipc_dataset_path)
        
        # Store best models for each modality
        self.best_models = {}
        
        print("🚀 Multimodal Legal AI System Initialized")
        print("=" * 60)
    
    def analyze_text_modality(self, text_input: str, true_sections: List[str] = None) -> Dict[str, Any]:
        """
        Analyze text modality with multiple models and select the best
        
        Args:
            text_input (str): Text input (case description, legal document, etc.)
            true_sections (List[str]): Ground truth IPC sections for evaluation
            
        Returns:
            Dict: Analysis results with best model selection
        """
        print("📝 Analyzing TEXT Modality")
        print("=" * 40)
        
        results = self.text_analyzer.analyze_with_all_models(text_input, true_sections)
        best_model = self._select_best_model(results, "text")
        
        return {
            'modality': 'text',
            'input': text_input,
            'best_model': best_model,
            'all_results': results,
            'comparison_metrics': self._generate_comparison_table(results)
        }
    
    def analyze_audio_modality(self, audio_path: str, true_sections: List[str] = None) -> Dict[str, Any]:
        """
        Analyze audio modality with multiple models and select the best
        
        Args:
            audio_path (str): Path to audio file
            true_sections (List[str]): Ground truth IPC sections for evaluation
            
        Returns:
            Dict: Analysis results with best model selection
        """
        print("🎵 Analyzing AUDIO Modality")
        print("=" * 40)
        
        results = self.audio_analyzer.analyze_with_all_models(audio_path, true_sections)
        best_model = self._select_best_model(results, "audio")
        
        return {
            'modality': 'audio',
            'input_path': audio_path,
            'best_model': best_model,
            'all_results': results,
            'comparison_metrics': self._generate_comparison_table(results)
        }
    
    def analyze_multimodal(self, inputs: Dict[str, str], true_sections: List[str] = None) -> Dict[str, Any]:
        """
        Analyze multiple modalities and combine results
        
        Args:
            inputs (Dict): Dictionary with modality as key and input as value
                Example: {
                    'text': 'case description',
                    'audio': 'path/to/audio.wav'
                }
            true_sections (List[str]): Ground truth IPC sections for evaluation
            
        Returns:
            Dict: Combined multimodal analysis results
        """
        print("🔄 Analyzing MULTIMODAL Input")
        print("=" * 50)
        
        results = {}
        combined_predictions = []
        
        # Analyze each modality
        for modality, input_data in inputs.items():
            if modality == 'text':
                result = self.analyze_text_modality(input_data, true_sections)
            elif modality == 'audio':
                result = self.analyze_audio_modality(input_data, true_sections)
            else:
                print(f"⚠️  Unknown modality: {modality}")
                continue
            
            results[modality] = result
            combined_predictions.extend(result['best_model'].predicted_sections)
        
        # Combine predictions from all modalities
        combined_analysis = self._combine_multimodal_predictions(combined_predictions, true_sections)
        
        return {
            'multimodal': True,
            'inputs': inputs,
            'modality_results': results,
            'combined_analysis': combined_analysis,
            'best_models_summary': self._get_best_models_summary()
        }
    
    def _select_best_model(self, results: List[ModelPerformance], modality: str) -> ModelPerformance:
        """
        Select the best model based on F1 score and inference time
        
        Args:
            results (List[ModelPerformance]): List of model performance results
            modality (str): Modality type
            
        Returns:
            ModelPerformance: Best performing model
        """
        if not results:
            return None
        
        # Sort by F1 score (primary) and inference time (secondary)
        sorted_results = sorted(results, key=lambda x: (x.f1_score, -x.inference_time), reverse=True)
        best_model = sorted_results[0]
        
        # Store best model for this modality
        self.best_models[modality] = best_model
        
        print(f"🏆 Best {modality.upper()} Model: {best_model.model_name}")
        print(f"   F1 Score: {best_model.f1_score:.3f}")
        print(f"   Inference Time: {best_model.inference_time:.2f}s")
        
        return best_model
    
    def _combine_multimodal_predictions(self, all_predictions: List[str], true_sections: List[str] = None) -> Dict[str, Any]:
        """
        Combine predictions from multiple modalities
        
        Args:
            all_predictions (List[str]): All predicted sections from different modalities
            true_sections (List[str]): Ground truth sections
            
        Returns:
            Dict: Combined analysis results
        """
        # Count frequency of each prediction
        from collections import Counter
        prediction_counts = Counter(all_predictions)
        
        # Get most common predictions
        most_common = prediction_counts.most_common(5)
        combined_sections = [section for section, count in most_common]
        
        # Calculate combined metrics if true sections provided
        combined_metrics = None
        if true_sections:
            combined_metrics = self.evaluation_metrics.calculate_metrics(
                true_sections, combined_sections
            )
        
        return {
            'combined_sections': combined_sections,
            'prediction_frequencies': dict(most_common),
            'metrics': combined_metrics,
            'confidence': len(set(combined_sections)) / len(combined_sections) if combined_sections else 0
        }
    
    def _generate_comparison_table(self, results: List[ModelPerformance]) -> pd.DataFrame:
        """
        Generate comparison table for model performance
        
        Args:
            results (List[ModelPerformance]): Model performance results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        data = []
        for result in results:
            data.append({
                'Model': result.model_name,
                'Accuracy': f"{result.accuracy:.3f}",
                'Precision': f"{result.precision:.3f}",
                'Recall': f"{result.recall:.3f}",
                'F1-Score': f"{result.f1_score:.3f}",
                'Inference Time (s)': f"{result.inference_time:.2f}",
                'Confidence': f"{result.confidence_score:.3f}"
            })
        
        return pd.DataFrame(data)
    
    def _get_best_models_summary(self) -> Dict[str, str]:
        """
        Get summary of best models for each modality
        
        Returns:
            Dict: Best model for each modality
        """
        summary = {}
        for modality, best_model in self.best_models.items():
            summary[modality] = {
                'model_name': best_model.model_name,
                'f1_score': best_model.f1_score,
                'inference_time': best_model.inference_time
            }
        return summary
    
    def run_comprehensive_evaluation(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on test cases
        
        Args:
            test_cases (List[Dict]): List of test cases with different modalities
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        print("🧪 Running Comprehensive Evaluation")
        print("=" * 50)
        
        evaluation_results = {
            'text': [],
            'audio': [],
            'multimodal': []
        }
        
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1}/{len(test_cases)} ---")
            
            # Analyze each modality if present
            for modality in ['text', 'audio']:
                if modality in test_case:
                    if modality == 'text':
                        result = self.analyze_text_modality(
                            test_case[modality], 
                            test_case.get('true_sections')
                        )
                    elif modality == 'audio':
                        result = self.analyze_audio_modality(
                            test_case[modality], 
                            test_case.get('true_sections')
                        )
                    # Append only the best_model (ModelPerformance object)
                    evaluation_results[modality].append(result['best_model'])
            
            # Analyze multimodal if multiple modalities present
            if len([k for k in test_case.keys() if k in ['text', 'audio']]) > 1:
                multimodal_inputs = {k: v for k, v in test_case.items() 
                                   if k in ['text', 'audio']}
                result = self.analyze_multimodal(multimodal_inputs, test_case.get('true_sections'))
                # For multimodal, append the combined_analysis dict (or None)
                evaluation_results['multimodal'].append(result['combined_analysis'])
        
        # Generate final summary
        final_summary = self._generate_evaluation_summary(evaluation_results)
        
        return {
            'evaluation_results': evaluation_results,
            'final_summary': final_summary,
            'best_models': self.best_models
        }
    
    def _generate_evaluation_summary(self, evaluation_results: Dict) -> Dict[str, Any]:
        """
        Generate final evaluation summary
        
        Args:
            evaluation_results (Dict): Results from comprehensive evaluation
            
        Returns:
            Dict: Final summary
        """
        summary = {}
        
        for modality, results in evaluation_results.items():
            if not results:
                continue
            # For multimodal, skip if not ModelPerformance
            if modality == 'multimodal':
                continue
            # Calculate average metrics
            avg_f1 = np.mean([r.f1_score for r in results if r])
            avg_time = np.mean([r.inference_time for r in results if r])
            best_model_obj = self.best_models.get(modality)
            summary[modality] = {
                'num_test_cases': len(results),
                'avg_f1_score': avg_f1,
                'avg_inference_time': avg_time,
                'best_model': best_model_obj.model_name if best_model_obj else 'N/A'
            }
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize multimodal system
    multimodal_ai = MultimodalLegalAI()
    
    # Test cases
    test_cases = [
        {
            'text': 'A person broke into a house and stole jewelry',
            'true_sections': ['IPC_380', 'IPC_457']
        },
        {
            'text': 'Someone killed another person in a fight',
            'true_sections': ['IPC_302']
        }
    ]
    
    # Run evaluation
    results = multimodal_ai.run_comprehensive_evaluation(test_cases)
    
    print("\n📊 Final Results:")
    print(json.dumps(results['final_summary'], indent=2)) 