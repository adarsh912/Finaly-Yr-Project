#!/usr/bin/env python3
"""
Basic Usage Examples for Legal AI System
Demonstrates single model and multimodal analysis capabilities
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.single_model_predictor import SingleModelPredictor
from core.multimodal_predictor import MultimodalPredictor

def example_single_model_analysis():
    """Example of single model text analysis"""
    print("🔍 Single Model Analysis Example")
    print("=" * 50)
    
    # Initialize single model predictor
    predictor = SingleModelPredictor(
        model_path="models/trained_model",
        ipc_dataset_path="data/ipc_sections.csv"
    )
    
    # Example legal case descriptions
    test_cases = [
        "The accused committed murder by shooting the victim with a firearm",
        "A person stole jewelry worth 50,000 rupees from a house",
        "The accused forged documents to obtain a loan fraudulently",
        "A person was found in possession of illegal drugs",
        "The accused committed rape and caused grievous hurt to the victim"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Case {i}: {case}")
        print("-" * 40)
        
        try:
            # Get predictions
            results = predictor.predict_text(case, top_k=3)
            
            # Display results
            for j, result in enumerate(results, 1):
                print(f"  {j}. Section {result['section']}: {result['confidence']:.3f}")
                print(f"     Description: {result['description'][:100]}...")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def example_multimodal_analysis():
    """Example of multimodal analysis"""
    print("\n🎯 Multimodal Analysis Example")
    print("=" * 50)
    
    # Initialize multimodal predictor
    predictor = MultimodalPredictor(
        model_path="models/trained_model",
        ipc_dataset_path="data/ipc_sections.csv"
    )
    
    # Example text analysis with ground truth
    test_text = "The accused entered a house at night and stole valuable items"
    true_sections = ["380", "454"]  # House-breaking and theft
    
    print(f"📝 Text Analysis: {test_text}")
    print(f"Ground Truth: Sections {true_sections}")
    print("-" * 40)
    
    try:
        results = predictor.analyze_text(test_text, true_sections=true_sections)
        
        for modality, modality_results in results.items():
            print(f"\n🎯 {modality.upper()} Analysis:")
            if modality_results:
                best_result = max(modality_results, key=lambda x: x.f1_score)
                print(f"  Best Model: {best_result.model_name}")
                print(f"  F1 Score: {best_result.f1_score:.3f}")
                print(f"  Precision: {best_result.precision:.3f}")
                print(f"  Recall: {best_result.recall:.3f}")
                print(f"  Inference Time: {best_result.inference_time:.2f}s")
                
                print("  Top Predictions:")
                for i, section in enumerate(best_result.predicted_sections[:3], 1):
                    confidence = best_result.confidence_scores[i-1] if i-1 < len(best_result.confidence_scores) else 0.0
                    print(f"    {i}. Section {section}: {confidence:.3f}")
            else:
                print("  ❌ No results available")
                
    except Exception as e:
        print(f"  ❌ Error: {e}")

def example_batch_processing():
    """Example of batch processing multiple inputs"""
    print("\n📦 Batch Processing Example")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SingleModelPredictor(
        model_path="models/trained_model",
        ipc_dataset_path="data/ipc_sections.csv"
    )
    
    # Batch of test cases
    test_cases = [
        "Murder case with multiple victims",
        "Financial fraud involving multiple parties",
        "Drug trafficking and money laundering",
        "Sexual assault and kidnapping",
        "Cybercrime and data theft"
    ]
    
    print("Processing batch of cases...")
    
    batch_results = {}
    for case in test_cases:
        try:
            results = predictor.predict_text(case, top_k=2)
            batch_results[case] = results
        except Exception as e:
            batch_results[case] = f"Error: {e}"
    
    # Display batch results
    for case, results in batch_results.items():
        print(f"\n📋 Case: {case[:50]}...")
        if isinstance(results, list):
            for i, result in enumerate(results, 1):
                print(f"  {i}. Section {result['section']}: {result['confidence']:.3f}")
        else:
            print(f"  ❌ {results}")

def example_performance_comparison():
    """Example of comparing different analysis modes"""
    print("\n⚖️ Performance Comparison Example")
    print("=" * 50)
    
    test_case = "The accused committed theft and caused grievous hurt to the victim"
    true_sections = ["379", "325"]  # Theft and grievous hurt
    
    print(f"Test Case: {test_case}")
    print(f"Ground Truth: Sections {true_sections}")
    print("-" * 40)
    
    # Single model analysis
    print("\n🔍 Single Model Analysis:")
    single_predictor = SingleModelPredictor(
        model_path="models/trained_model",
        ipc_dataset_path="data/ipc_sections.csv"
    )
    
    try:
        single_results = single_predictor.predict_text(test_case, top_k=5)
        print("  Predictions:")
        for i, result in enumerate(single_results, 1):
            print(f"    {i}. Section {result['section']}: {result['confidence']:.3f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Multimodal analysis
    print("\n🎯 Multimodal Analysis:")
    multimodal_predictor = MultimodalPredictor(
        model_path="models/trained_model",
        ipc_dataset_path="data/ipc_sections.csv"
    )
    
    try:
        multimodal_results = multimodal_predictor.analyze_text(test_case, true_sections=true_sections)
        
        for modality, modality_results in multimodal_results.items():
            if modality_results:
                best_result = max(modality_results, key=lambda x: x.f1_score)
                print(f"  {modality.upper()}:")
                print(f"    F1 Score: {best_result.f1_score:.3f}")
                print(f"    Precision: {best_result.precision:.3f}")
                print(f"    Recall: {best_result.recall:.3f}")
                print(f"    Time: {best_result.inference_time:.2f}s")
    except Exception as e:
        print(f"  ❌ Error: {e}")

def main():
    """Run all examples"""
    print("🏛️ Legal AI System - Basic Usage Examples")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/trained_model"
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at: {model_path}")
        print("Please ensure the trained model is available before running examples.")
        return
    
    # Check if IPC dataset exists
    ipc_path = "data/ipc_sections.csv"
    if not os.path.exists(ipc_path):
        print(f"❌ IPC dataset not found at: {ipc_path}")
        print("Please ensure the IPC dataset is available before running examples.")
        return
    
    try:
        # Run examples
        example_single_model_analysis()
        example_multimodal_analysis()
        example_batch_processing()
        example_performance_comparison()
        
        print("\n✅ All examples completed successfully!")
        print("\n💡 Tips:")
        print("- Use --verbose flag for detailed output")
        print("- Save results to file with --output flag")
        print("- Adjust top-k for different numbers of predictions")
        print("- Provide ground truth sections for evaluation")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 