import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

class ImprovedLegalPredictor:
    def __init__(self, model_path="trained_model"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 512
        self.threshold = 0.25  # Updated threshold
        self.max_predictions = 5  # Limit number of predictions
        
        # Load model components
        with open(os.path.join(model_path, 'classes.json'), 'r') as f:
            self.classes = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(self.classes),
            problem_type="multi_label_classification"
        )
        self.model.to(self.device)
        self.model.eval()
    
    def clean_text(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        elif not isinstance(text, str):
            text = str(text)
        
        text = ' '.join(text.split())
        text = re.sub(r'Section\s+(\d+)', r'Section\1', text)
        text = re.sub(r'\[\d+\]', '', text)
        return text
    
    def predict(self, text):
        # Clean and prepare text
        text = self.clean_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        
        # Convert to numpy
        probabilities = probabilities.cpu().numpy().flatten()
        
        # Get top predictions above threshold
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > self.threshold:
                predictions.append({
                    'section': self.classes[i],
                    'confidence': prob
                })
        
        # Sort by confidence and limit to max_predictions
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        predictions = predictions[:self.max_predictions]
        
        return {
            'text': text[:200] + "..." if len(text) > 200 else text,
            'predictions': predictions,
            'total_sections_checked': len(self.classes)
        }

# Usage example
if __name__ == "__main__":
    predictor = ImprovedLegalPredictor()
    
    test_cases = [
        "The accused murdered the victim by stabbing them with a knife.",
        "The accused stole a gold necklace worth 25,000 rupees from the victim.",
        "The accused assaulted the victim causing bodily injury."
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}: {case}")
        result = predictor.predict(case)
        print(f"Predictions: {result['predictions']}")
