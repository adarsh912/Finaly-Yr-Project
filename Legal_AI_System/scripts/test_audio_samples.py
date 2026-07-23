#!/usr/bin/env python3
"""
Audio Samples Test Script
Tests audio processing with the organized audio samples
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modalities.audio_modality import AudioModalityAnalyzer
from core.multimodal_predictor import MultimodalPredictor

def load_audio_metadata(samples_dir: str) -> dict:
    """Load audio samples metadata"""
    metadata_path = os.path.join(samples_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {"audio_samples": []}

def test_audio_samples(samples_dir: str, mode: str = "multimodal", top_k: int = 5):
    """Test audio processing with all available samples"""
    
    print("🎵 Audio Samples Test")
    print("=" * 50)
    
    # Load metadata
    metadata = load_audio_metadata(samples_dir)
    audio_samples = metadata.get("audio_samples", [])
    
    if not audio_samples:
        print("❌ No audio samples found in metadata")
        return
    
    print(f"📁 Found {len(audio_samples)} audio samples")
    print(f"📂 Samples directory: {samples_dir}")
    print()
    
    # Initialize predictor based on mode
    if mode == "multimodal":
        predictor = MultimodalPredictor(
            model_path="models/trained_model",
            ipc_dataset_path="data/ipc_sections.csv"
        )
    else:
        from core.single_model_predictor import SingleModelPredictor
        # Create config for SingleModelPredictor
        config = {
            'model_path': 'models/trained_model',
            'ipc_dataset_path': 'data/ipc_sections.csv',
            'threshold': 0.25,
            'max_predictions': top_k,
            'max_length': 512
        }
        predictor = SingleModelPredictor(config)
    
    # Process each audio sample
    for i, sample in enumerate(audio_samples, 1):
        filename = sample["filename"]
        audio_path = os.path.join(samples_dir, filename)
        
        if not os.path.exists(audio_path):
            print(f"⚠️  Audio file not found: {audio_path}")
            continue
        
        print(f"\n🎵 Processing Sample {i}: {filename}")
        print(f"📄 Description: {sample.get('description', 'N/A')}")
        print(f"📊 Size: {sample.get('size_bytes', 0)} bytes")
        print("-" * 40)
        
        try:
            if mode == "multimodal":
                # Multimodal analysis
                results = predictor.analyze_audio(audio_path)
                
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
                        for j, section in enumerate(best_result.predicted_sections[:top_k], 1):
                            confidence = best_result.confidence_scores[j-1] if j-1 < len(best_result.confidence_scores) else 0.0
                            print(f"  {j}. Section {section}: {confidence:.3f}")
                    else:
                        print("❌ No audio analysis results")
                else:
                    print("❌ No results from multimodal analysis")
            else:
                # Single model analysis
                results = predictor.predict_audio(audio_path, top_k=top_k)
                
                print("🔍 Predictions:")
                for j, result in enumerate(results, 1):
                    print(f"  {j}. Section {result['section']}: {result['confidence']:.3f}")
                    print(f"     Description: {result['description']}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
        
        print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Audio Samples Processing")
    parser.add_argument("--samples-dir", default="data/audio_samples",
                       help="Directory containing audio samples")
    parser.add_argument("--mode", choices=["single", "multimodal"], default="multimodal",
                       help="Analysis mode")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top predictions to show")
    
    args = parser.parse_args()
    
    # Check if samples directory exists
    if not os.path.exists(args.samples_dir):
        print(f"❌ Samples directory not found: {args.samples_dir}")
        return
    
    # Run the test
    test_audio_samples(args.samples_dir, args.mode, args.top_k)
    
    print("\n✅ Audio samples test completed!")

if __name__ == "__main__":
    main() 