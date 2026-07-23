#!/usr/bin/env python3
"""
Multi-ASR Test Script
Tests multiple LLM-based ASR models and compares their results
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modalities.multi_asr_processor import MultiASRProcessor
from core.single_model_predictor import SingleModelPredictor

def test_multi_asr(audio_path: str, config: dict = None):
    """Test multiple ASR models on audio file"""
    
    print("🎤 Multi-ASR Test")
    print("=" * 50)
    print(f"Audio file: {audio_path}")
    print()
    
    # Initialize multi-ASR processor
    asr_processor = MultiASRProcessor(config)
    
    # Get available models
    available_models = asr_processor.get_available_models()
    print(f"📋 Available ASR Models: {', '.join(available_models)}")
    print()
    
    # Transcribe with all models
    print("🔄 Transcribing with all models...")
    results = asr_processor.transcribe_audio(audio_path)
    
    if not results:
        print("❌ No transcription results")
        return
    
    # Display individual results
    print("\n📊 Individual Results:")
    print("-" * 40)
    
    for result in results:
        print(f"\n🎯 {result.model_name}:")
        print(f"   Text: {result.text[:100]}{'...' if len(result.text) > 100 else ''}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Processing Time: {result.processing_time:.2f}s")
        if hasattr(result, 'error'):
            print(f"   Error: {result.error}")
    
    # Compare results
    print("\n🔍 Comparison Analysis:")
    print("-" * 40)
    
    comparison = asr_processor.compare_results(results)
    
    print(f"Total Models: {comparison['total_models']}")
    print(f"Best Model: {comparison.get('best_model', 'None')}")
    if comparison.get('best_text'):
        print(f"Best Text: {comparison['best_text'][:100]}{'...' if len(comparison['best_text']) > 100 else ''}")
        print(f"Best Confidence: {comparison.get('best_confidence', 0):.3f}")
    
    print(f"Unique Text Variations: {comparison['unique_texts']}")
    
    # Processing times comparison
    print("\n⏱️  Processing Times:")
    for model, time_taken in comparison['processing_times'].items():
        print(f"   {model}: {time_taken:.2f}s")
    
    return results, comparison

def test_legal_classification(asr_results, ipc_dataset_path: str):
    """Test legal classification on ASR results"""
    
    print("\n⚖️  Legal Classification Test:")
    print("-" * 40)
    
    # Initialize legal classifier
    config = {
        'model_path': 'models/trained_model',
        'ipc_dataset_path': ipc_dataset_path,
        'threshold': 0.25,
        'max_predictions': 5,
        'max_length': 512
    }
    
    predictor = SingleModelPredictor(config)
    
    # Classify each ASR result
    classification_results = {}
    
    for result in asr_results:
        if not result.text or hasattr(result, 'error'):
            continue
            
        print(f"\n🔍 Classifying {result.model_name} result:")
        print(f"   Text: {result.text[:80]}{'...' if len(result.text) > 80 else ''}")
        
        try:
            # Get legal predictions
            legal_result = predictor.predict_text(result.text)
            predictions = legal_result.get('predictions', [])
            
            classification_results[result.model_name] = {
                'text': result.text,
                'predictions': predictions,
                'confidence': result.confidence,
                'processing_time': result.processing_time
            }
            
            print("   Top Predictions:")
            for i, pred in enumerate(predictions[:3], 1):
                print(f"     {i}. Section {pred['section']}: {pred['confidence']:.3f}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return classification_results

def compare_legal_results(classification_results):
    """Compare legal classification results from different ASR models"""
    
    print("\n📈 Legal Classification Comparison:")
    print("-" * 40)
    
    if not classification_results:
        print("❌ No classification results to compare")
        return
    
    # Compare predictions
    all_sections = set()
    for model, result in classification_results.items():
        sections = [pred['section'] for pred in result['predictions']]
        all_sections.update(sections)
    
    print(f"Total unique sections predicted: {len(all_sections)}")
    
    # Show predictions by model
    for model, result in classification_results.items():
        print(f"\n🎯 {model}:")
        print(f"   ASR Confidence: {result['confidence']:.3f}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        print("   Top Predictions:")
        for i, pred in enumerate(result['predictions'][:3], 1):
            print(f"     {i}. Section {pred['section']}: {pred['confidence']:.3f}")
    
    # Find best overall result
    best_model = max(classification_results.keys(), 
                    key=lambda x: classification_results[x]['confidence'])
    
    print(f"\n🏆 Best Overall Result: {best_model}")
    print(f"   ASR Confidence: {classification_results[best_model]['confidence']:.3f}")
    print(f"   Text: {classification_results[best_model]['text'][:100]}{'...' if len(classification_results[best_model]['text']) > 100 else ''}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Multi-ASR System")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--config", help="Path to ASR configuration file")
    parser.add_argument("--ipc-dataset", default="data/ipc_sections.csv", 
                       help="Path to IPC dataset")
    parser.add_argument("--legal-only", action="store_true", 
                       help="Skip ASR comparison, only do legal classification")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        return
    
    # Load configuration if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    if not args.legal_only:
        # Test ASR models
        asr_results, comparison = test_multi_asr(args.audio, config)
        
        if not asr_results:
            return
        
        # Test legal classification
        classification_results = test_legal_classification(asr_results, args.ipc_dataset)
        
        # Compare legal results
        compare_legal_results(classification_results)
    else:
        # Only test legal classification with a single ASR result
        print("⚖️  Legal Classification Only Test")
        print("=" * 50)
        
        # Use a simple ASR result for testing
        from modalities.multi_asr_processor import ASRResult
        test_result = ASRResult(
            text="The accused committed theft by taking property without permission",
            confidence=0.9,
            processing_time=1.0,
            model_name="Test ASR"
        )
        
        classification_results = test_legal_classification([test_result], args.ipc_dataset)
        compare_legal_results(classification_results)
    
    print("\n✅ Multi-ASR test completed!")

if __name__ == "__main__":
    main() 