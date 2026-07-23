#!/usr/bin/env python3
"""
Legal AI System - Main Entry Point
Unified system for multi-label classification of Indian legal documents using custom trained BERT model.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.multimodal_predictor import MultimodalPredictor


def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return {
        "ipc_dataset_path": "data/ipc_sections.csv",
        "model_path": "models/trained_model",
        "output_dir": "outputs",
        "log_level": "INFO",
        "use_gpu": True,
        "batch_size": 32,
        "max_length": 512,
        "top_k": 5
    }


def main():
    """Main entry point for the Legal AI System."""
    parser = argparse.ArgumentParser(description="Legal AI System - Multi-label IPC Classification")
    parser.add_argument("--input", required=True, help="Input file or text")
    parser.add_argument("--input-type", choices=["text", "audio", "file"], default="text", help="Type of input")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--true-sections", nargs="*", help="Ground truth sections for evaluation")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config["output_dir"], exist_ok=True)

    print("🏛️  Legal AI System - Multi-label IPC Classification")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Input Type: {args.input_type}")
    print(f"Model: {config['model_path']}")
    print("=" * 60)

    try:
        predictor = MultimodalPredictor(
            config["model_path"],
            config["ipc_dataset_path"]
        )

        if args.input_type == "text":
            results = predictor.analyze_text(args.input, true_sections=args.true_sections)
        elif args.input_type == "audio":
            results = predictor.analyze_audio(args.input, true_sections=args.true_sections)
        elif args.input_type == "file":
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            results = predictor.analyze_text(text, true_sections=args.true_sections)
        else:
            print(f"❌ Input type '{args.input_type}' not supported")
            return

        print("\n📊 Multimodal Analysis Results:")
        print("-" * 50)
        if results:
            print("\n🎯 MULTIMODAL Analysis:")
            print("-" * 30)
            if 'model_name' in results:
                print(f"Best Model: {results['model_name']}")
            if 'f1_score' in results:
                print(f"F1 Score: {results['f1_score']:.3f}")
            if 'confidence' in results:
                print(f"Confidence: {results['confidence']:.3f}")
            print("\nTop Predictions:")
            for i, pred in enumerate(results.get('predictions', [])[:args.top_k], 1):
                print(f"  {i}. Section {pred['section']}: {pred.get('confidence', 0.0):.3f}")
        else:
            print("❌ No results available")

        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            output_data = {
                "mode": "multimodal",
                "input": args.input,
                "input_type": args.input_type,
                "results": results
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")

        print("\n✅ Analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
