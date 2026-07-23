#!/usr/bin/env python3
"""
Evaluation Metrics Module
Provides comprehensive evaluation metrics for multimodal legal AI system
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

class EvaluationMetrics:
    """Comprehensive evaluation metrics for legal AI system"""
    
    def __init__(self):
        """Initialize evaluation metrics"""
        pass
    
    def calculate_metrics(self, true_sections: List[str], predicted_sections: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for IPC section prediction
        
        Args:
            true_sections (List[str]): Ground truth IPC sections
            predicted_sections (List[str]): Predicted IPC sections
            
        Returns:
            Dict[str, float]: Dictionary containing all metrics
        """
        if not true_sections and not predicted_sections:
            return {
                'accuracy': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'exact_match': 1.0,
                'partial_match': 1.0
            }
        
        # Convert to sets for set-based metrics
        true_set = set(true_sections)
        pred_set = set(predicted_sections)
        
        # Basic metrics
        accuracy = self._calculate_accuracy(true_set, pred_set)
        precision = self._calculate_precision(true_set, pred_set)
        recall = self._calculate_recall(true_set, pred_set)
        f1 = self._calculate_f1_score(precision, recall)
        
        # Additional metrics
        exact_match = self._calculate_exact_match(true_set, pred_set)
        partial_match = self._calculate_partial_match(true_set, pred_set)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'exact_match': exact_match,
            'partial_match': partial_match
        }
    
    def _calculate_accuracy(self, true_set: set, pred_set: set) -> float:
        """Calculate accuracy (Jaccard similarity)"""
        if not true_set and not pred_set:
            return 1.0
        elif not true_set or not pred_set:
            return 0.0
        
        intersection = len(true_set.intersection(pred_set))
        union = len(true_set.union(pred_set))
        return intersection / union if union > 0 else 0.0
    
    def _calculate_precision(self, true_set: set, pred_set: set) -> float:
        """Calculate precision"""
        if not pred_set:
            return 1.0 if not true_set else 0.0
        
        intersection = len(true_set.intersection(pred_set))
        return intersection / len(pred_set)
    
    def _calculate_recall(self, true_set: set, pred_set: set) -> float:
        """Calculate recall"""
        if not true_set:
            return 1.0 if not pred_set else 0.0
        
        intersection = len(true_set.intersection(pred_set))
        return intersection / len(true_set)
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_exact_match(self, true_set: set, pred_set: set) -> float:
        """Calculate exact match (perfect prediction)"""
        return 1.0 if true_set == pred_set else 0.0
    
    def _calculate_partial_match(self, true_set: set, pred_set: set) -> float:
        """Calculate partial match (at least one correct prediction)"""
        if not true_set:
            return 1.0 if not pred_set else 0.0
        
        intersection = len(true_set.intersection(pred_set))
        return 1.0 if intersection > 0 else 0.0
    
    def calculate_model_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregate performance metrics across multiple test cases
        
        Args:
            results (List[Dict]): List of results from multiple test cases
            
        Returns:
            Dict[str, Any]: Aggregate performance metrics
        """
        if not results:
            return {}
        
        # Extract metrics from all results
        metrics_list = []
        for result in results:
            if 'metrics' in result and result['metrics']:
                metrics_list.append(result['metrics'])
        
        if not metrics_list:
            return {}
        
        # Calculate averages
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'exact_match', 'partial_match']:
            values = [m.get(metric, 0.0) for m in metrics_list]
            avg_metrics[f'avg_{metric}'] = np.mean(values)
            avg_metrics[f'std_{metric}'] = np.std(values)
            avg_metrics[f'min_{metric}'] = np.min(values)
            avg_metrics[f'max_{metric}'] = np.max(values)
        
        # Additional statistics
        avg_metrics['num_test_cases'] = len(metrics_list)
        avg_metrics['success_rate'] = np.mean([m.get('exact_match', 0.0) for m in metrics_list])
        
        return avg_metrics
    
    def generate_comparison_report(self, model_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report for multiple models
        
        Args:
            model_results (Dict[str, List[Dict]]): Results for different models
            
        Returns:
            Dict[str, Any]: Comparison report
        """
        comparison = {}
        
        for model_name, results in model_results.items():
            if results:
                performance = self.calculate_model_performance(results)
                comparison[model_name] = performance
        
        # Find best model for each metric
        best_models = {}
        for metric in ['avg_f1_score', 'avg_accuracy', 'avg_precision', 'avg_recall']:
            best_model = None
            best_score = -1
            
            for model_name, performance in comparison.items():
                score = performance.get(metric, 0.0)
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            best_models[metric] = {
                'model': best_model,
                'score': best_score
            }
        
        comparison['best_models'] = best_models
        
        return comparison
    
    def calculate_confidence_metrics(self, predictions: List[Dict]) -> Dict[str, float]:
        """
        Calculate confidence-related metrics
        
        Args:
            predictions (List[Dict]): List of predictions with confidence scores
            
        Returns:
            Dict[str, float]: Confidence metrics
        """
        if not predictions:
            return {}
        
        confidence_scores = [pred.get('Score', 0.0) for pred in predictions]
        
        return {
            'avg_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores),
            'high_confidence_rate': np.mean([1.0 if c > 0.7 else 0.0 for c in confidence_scores])
        }
    
    def calculate_diversity_metrics(self, predictions: List[Dict]) -> Dict[str, float]:
        """
        Calculate diversity metrics for predictions
        
        Args:
            predictions (List[Dict]): List of predictions
            
        Returns:
            Dict[str, float]: Diversity metrics
        """
        if not predictions:
            return {}
        
        sections = [pred.get('Section', '') for pred in predictions]
        unique_sections = set(sections)
        
        return {
            'num_unique_sections': len(unique_sections),
            'diversity_ratio': len(unique_sections) / len(sections) if sections else 0.0,
            'prediction_variety': len(unique_sections)
        }
    
    def generate_detailed_report(self, true_sections: List[str], predictions: List[Dict]) -> Dict[str, Any]:
        """
        Generate detailed evaluation report
        
        Args:
            true_sections (List[str]): Ground truth sections
            predictions (List[Dict]): Model predictions
            
        Returns:
            Dict[str, Any]: Detailed report
        """
        predicted_sections = [pred.get('Section', '') for pred in predictions]
        
        # Basic metrics
        basic_metrics = self.calculate_metrics(true_sections, predicted_sections)
        
        # Confidence metrics
        confidence_metrics = self.calculate_confidence_metrics(predictions)
        
        # Diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(predictions)
        
        # Detailed analysis
        analysis = {
            'true_sections': true_sections,
            'predicted_sections': predicted_sections,
            'correct_predictions': list(set(true_sections).intersection(set(predicted_sections))),
            'missed_sections': list(set(true_sections) - set(predicted_sections)),
            'incorrect_predictions': list(set(predicted_sections) - set(true_sections))
        }
        
        return {
            'basic_metrics': basic_metrics,
            'confidence_metrics': confidence_metrics,
            'diversity_metrics': diversity_metrics,
            'analysis': analysis
        }

# Example usage
if __name__ == "__main__":
    # Initialize evaluation metrics
    evaluator = EvaluationMetrics()
    
    # Test metrics calculation
    true_sections = ["IPC_380", "IPC_457"]
    predicted_sections = ["IPC_380", "IPC_392", "IPC_457"]
    
    metrics = evaluator.calculate_metrics(true_sections, predicted_sections)
    print("Metrics:", metrics)
    
    # Test detailed report
    predictions = [
        {'Section': 'IPC_380', 'Score': 0.9},
        {'Section': 'IPC_392', 'Score': 0.7},
        {'Section': 'IPC_457', 'Score': 0.8}
    ]
    
    report = evaluator.generate_detailed_report(true_sections, predictions)
    print("Detailed Report:", report) 