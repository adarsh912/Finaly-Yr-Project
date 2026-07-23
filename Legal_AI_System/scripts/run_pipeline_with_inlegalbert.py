#!/usr/bin/env python3
"""
Pipeline: Text Description -> LLM/Transformer (paraphrase) -> Custom Trained BERT Model -> IPC Prediction & Metrics
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from modalities.text_modality import TextModalityAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline: text -> LLM/transformer -> Custom Trained BERT Model -> IPC prediction & metrics.")
    parser.add_argument('--input', type=str, required=True, help='Input text description')
    parser.add_argument('--truth', type=str, required=True, help='Comma-separated ground truth IPC sections (e.g. IPC_380,IPC_457)')
    parser.add_argument('--ipc_dataset', type=str, default='ipc_sections.csv', help='Path to IPC dataset')
    parser.add_argument('--output', type=str, default='results/pipeline_bert_text.csv', help='Path to save results CSV')
    return parser.parse_args()

def paraphrase_with_model(analyzer, text, model_name):
    """Use a model to paraphrase or enhance the input text."""
    model_info = analyzer.models[model_name]
    if model_info['type'] == 'hf_pipeline':
        # Use text-generation pipeline for paraphrasing
        pipeline_model = model_info['pipeline']
        try:
            paraphrased = pipeline_model(text, max_length=128, num_return_sequences=1)[0]['generated_text']
            return paraphrased
        except Exception as e:
            print(f"⚠️  Paraphrasing with {model_name} failed: {e}")
            return text
    elif model_info['type'] == 'openai':
        # Use OpenAI GPT for paraphrasing
        import openai
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Paraphrase the following legal case description in clear, formal English."},
                    {"role": "user", "content": text}
                ],
                max_tokens=128
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  OpenAI GPT paraphrasing failed: {e}")
            return text
    else:
        # For transformer models, just return the original text (or optionally add more logic)
        return text

def main():
    args = parse_args()
    true_sections = [s.strip() for s in args.truth.split(',') if s.strip()]
    analyzer = TextModalityAnalyzer(args.ipc_dataset)

    print(f"\n=== Pipeline: Text -> LLM/Transformer -> Custom Trained BERT Model ===")
    results = []
    # For each model, paraphrase/enhance the input, then feed to Custom Trained BERT Model
    for model_name in tqdm(analyzer.models.keys(), desc="Testing LLM/transformers as preprocessors"):
        print(f"\n--- Using {model_name} ---")
        paraphrased_text = paraphrase_with_model(analyzer, args.input, model_name)
        print(f"Paraphrased/Enhanced Text: {paraphrased_text}")
        # Feed to Custom Trained BERT Model
        bert_result = analyzer.analyze_with_model(paraphrased_text, 'Custom Trained BERT', true_sections)
        print(f"Predicted Sections: {bert_result.predicted_sections}")
        print(f"F1: {bert_result.f1_score:.3f}, Precision: {bert_result.precision:.3f}, Recall: {bert_result.recall:.3f}, Accuracy: {bert_result.accuracy:.3f}, Time: {bert_result.inference_time:.2f}s")
        row = {
            'input_text': args.input,
            'model_used': model_name,
            'bert_predicted_sections': ','.join(bert_result.predicted_sections),
            'true_sections': ','.join(bert_result.true_sections),
            'f1_score': bert_result.f1_score,
            'precision': bert_result.precision,
            'recall': bert_result.recall,
            'accuracy': bert_result.accuracy,
            'inference_time': bert_result.inference_time,
            'confidence': bert_result.confidence_score
        }
        results.append(row)
    # Save results
    df = pd.DataFrame(results)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 