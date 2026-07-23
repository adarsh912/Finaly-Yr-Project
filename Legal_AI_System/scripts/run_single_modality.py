#!/usr/bin/env python3
"""
Run Single Modality Analysis
Test individual modality analyzers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modalities.text_modality import TextModalityAnalyzer
from modalities.audio_modality import AudioModalityAnalyzer

def test_text_modality():
    """Test text modality analyzer"""
    print("Testing Text Modality...")
    
    # Sample legal text
    text = "The accused committed theft by taking property without permission from the victim's house."
    
    analyzer = TextModalityAnalyzer('data/ipc_sections.csv')
    results = analyzer.analyze_with_all_models(text)
    
    print(f"Found {len(results)} results")
    for result in results:
        print(f"Model: {result.model_name}")
        print(f"Predicted sections: {result.predicted_sections}")
        print(f"F1 Score: {result.f1_score:.3f}")
        print()

def test_audio_modality():
    """Test audio modality analyzer"""
    print("Testing Audio Modality...")
    
    # Test with audio file
    audio_path = "data/audio_samples/audio1.mp3"
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
    
    analyzer = AudioModalityAnalyzer('data/ipc_sections.csv')
    results = analyzer.analyze_with_all_models(audio_path)
    
    print(f"Found {len(results)} results")
    for result in results:
        print(f"Model: {result.model_name}")
        print(f"Predicted sections: {result.predicted_sections}")
        print(f"F1 Score: {result.f1_score:.3f}")
        print()

if __name__ == "__main__":
    print("Running Single Modality Tests")
    print("=" * 40)
    
    test_text_modality()
    test_audio_modality()
    
    print("Tests completed!") 