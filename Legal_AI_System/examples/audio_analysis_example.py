#!/usr/bin/env python3
"""
Audio Analysis Example
Demonstrates how to use the Legal AI System with audio samples
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.multimodal_predictor import MultimodalPredictor
from core.single_model_predictor import SingleModelPredictor

def example_audio_analysis():
    """Example of audio analysis using the Legal AI System"""
    
    print("🎵 Audio Analysis Example")
    print("=" * 50)
    
    # Configuration
    model_path = "models/trained_model"
    ipc_dataset_path = "data/ipc_sections.csv"
    audio_samples_dir = "data/audio_samples"
    
    # Check if audio samples exist
    if not os.path.exists(audio_samples_dir):
        print(f"❌ Audio samples directory not found: {audio_samples_dir}")
        return
    
    # List available audio files
    audio_files = [f for f in os.listdir(audio_samples_dir) if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))]
    
    if not audio_files:
        print("❌ No audio files found in samples directory")
        return
    
    print(f"📁 Found {len(audio_files)} audio files:")
    for audio_file in audio_files:
        print(f"  - {audio_file}")
    print()
    
    # Example 1: Single Model Analysis
    print("🔍 Example 1: Single Model Audio Analysis")
    print("-" * 40)
    
    single_predictor = SingleModelPredictor(
        model_path=model_path,
        ipc_dataset_path=ipc_dataset_path
    )
    
    # Process first audio file
    first_audio = os.path.join(audio_samples_dir, audio_files[0])
    print(f"Processing: {audio_files[0]}")
    
    try:
        results = single_predictor.predict_audio(first_audio, top_k=3)
        
        print("📊 Predictions:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Section {result['section']}: {result['confidence']:.3f}")
            print(f"     Description: {result['description']}")
        
    except Exception as e:
        print(f"❌ Error in single model analysis: {e}")
    
    print()
    
    # Example 2: Multimodal Analysis
    print("🎯 Example 2: Multimodal Audio Analysis")
    print("-" * 40)
    
    multimodal_predictor = MultimodalPredictor(
        model_path=model_path,
        ipc_dataset_path=ipc_dataset_path
    )
    
    # Process second audio file (if available)
    if len(audio_files) > 1:
        second_audio = os.path.join(audio_samples_dir, audio_files[1])
        print(f"Processing: {audio_files[1]}")
        
        try:
            results = multimodal_predictor.analyze_audio(second_audio)
            
            if results and "audio" in results:
                audio_results = results["audio"]
                if audio_results:
                    best_result = max(audio_results, key=lambda x: x.f1_score)
                    print(f"✅ Best Model: {best_result.model_name}")
                    print(f"📈 F1 Score: {best_result.f1_score:.3f}")
                    print(f"🎯 Precision: {best_result.precision:.3f}")
                    print(f"📊 Recall: {best_result.recall:.3f}")
                    print(f"⏱️  Inference Time: {best_result.inference_time:.2f}s")
                    
                    print("\n🔍 Top Predictions:")
                    for i, section in enumerate(best_result.predicted_sections[:3], 1):
                        confidence = best_result.confidence_scores[i-1] if i-1 < len(best_result.confidence_scores) else 0.0
                        print(f"  {i}. Section {section}: {confidence:.3f}")
                else:
                    print("❌ No audio analysis results")
            else:
                print("❌ No results from multimodal analysis")
                
        except Exception as e:
            print(f"❌ Error in multimodal analysis: {e}")
    
    print()
    
    # Example 3: Batch Processing
    print("📦 Example 3: Batch Audio Processing")
    print("-" * 40)
    
    print("Processing all audio files with single model...")
    
    for audio_file in audio_files[:2]:  # Process first 2 files
        audio_path = os.path.join(audio_samples_dir, audio_file)
        print(f"\n🎵 {audio_file}:")
        
        try:
            results = single_predictor.predict_audio(audio_path, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Section {result['section']}: {result['confidence']:.3f}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n✅ Audio analysis examples completed!")

def example_cli_usage():
    """Example CLI commands for audio analysis"""
    
    print("\n💻 CLI Usage Examples")
    print("=" * 30)
    
    print("Single model analysis:")
    print("python main.py --mode single --input data/audio_samples/audio1.mp3 --input-type audio --top-k 3")
    print()
    
    print("Multimodal analysis:")
    print("python main.py --mode multimodal --input data/audio_samples/audio2.mp3 --input-type audio --top-k 5")
    print()
    
    print("Test audio samples script:")
    print("python scripts/test_audio_samples.py --mode multimodal --top-k 3")
    print()

if __name__ == "__main__":
    example_audio_analysis()
    example_cli_usage() 