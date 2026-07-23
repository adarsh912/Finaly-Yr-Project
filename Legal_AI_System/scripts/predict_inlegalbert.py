import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# Define paths
MODEL_PATH = "trained_model"
DATASET_PATH = "Dataset"

class Config:
    """
    Configuration class for prediction settings
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 512
        self.threshold = 0.25

config = Config()

def verify_model_files():
    """
    Verify that all required model files exist
    """
    print("🔍 Checking model files...")
    required_files = [
        "config.json",
        "model.safetensors", 
        "tokenizer.json",
        "classes.json",
        "test_metrics.json"
    ]
    
    all_exist = True
    for file in required_files:
        filepath = os.path.join(MODEL_PATH, file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024*1024)  # MB
            print(f"  ✅ {file} ({size:.1f} MB)")
        else:
            print(f"  ❌ {file} - MISSING")
            all_exist = False
    
    if all_exist:
        print("✅ All model files verified successfully!")
    else:
        print("❌ Some model files are missing!")
    
    return all_exist

def display_model_metrics():
    """
    Load and display model performance metrics
    """
    print(f"\n📊 Loading test metrics...")
    try:
        with open(os.path.join(MODEL_PATH, "test_metrics.json"), 'r') as f:
            metrics = json.load(f)
        
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        
        return metrics
    except Exception as e:
        print(f"  ❌ Error loading metrics: {e}")
        return None

def clean_legal_text(text):
    """
    Clean and standardize legal text
    """
    if isinstance(text, list):
        text = ' '.join(text)
    elif not isinstance(text, str):
        text = str(text)
    
    text = ' '.join(text.split())
    text = re.sub(r'Section\s+(\d+)', r'Section\1', text)
    text = re.sub(r'\[\d+\]', '', text)
    return text

def load_trained_model():
    """
    Load the trained model, tokenizer, and class mapping
    """
    print("Loading trained model...")
    
    # Load class mapping
    with open(os.path.join(MODEL_PATH, 'classes.json'), 'r') as f:
        classes = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(classes),
        problem_type="multi_label_classification"
    )
    
    model.to(config.device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Number of classes: {len(classes)}")
    print(f"Device: {config.device}")
    
    return model, tokenizer, classes

def predict_single_text(model, tokenizer, text, classes, threshold=0.25):
    """
    Make prediction for a single text input
    """
    # Clean and prepare text
    text = clean_legal_text(text)
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=config.max_length,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
    
    # Convert to numpy
    probabilities = probabilities.cpu().numpy().flatten()
    predictions = predictions.cpu().numpy().flatten()
    
    # Get predicted classes
    predicted_classes = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            predicted_classes.append(classes[i])
    
    return {
        'probabilities': probabilities,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'text_preview': text[:200] + "..." if len(text) > 200 else text
    }

def predict_custom_test_cases(model, tokenizer, classes):
    """
    Make predictions on custom test cases
    """
    print("\n" + "="*60)
    print("CUSTOM TEST CASES PREDICTION")
    print("="*60)
    
    # Custom test cases with different legal scenarios
    test_cases = [
        {
            "name": "Murder Case",
            "text": "The accused intentionally caused the death of the victim by stabbing them multiple times with a knife. The act was premeditated and carried out with the intention to kill."
        },
        {
            "name": "Theft Case", 
            "text": "The defendant unlawfully took possession of valuable jewelry from the victim's house without their consent. The stolen items were worth approximately 50,000 rupees."
        },
        {
            "name": "Assault Case",
            "text": "The accused assaulted the victim by punching them in the face, causing bodily injury. The attack was unprovoked and resulted in the victim requiring medical treatment."
        },
        {
            "name": "Fraud Case",
            "text": "The defendant deceived the complainant by making false representations about a property deal. They received money under false pretenses and failed to deliver the promised property."
        },
        {
            "name": "Traffic Violation",
            "text": "The accused was driving a vehicle at a speed exceeding the legal limit in a residential area, endangering the safety of pedestrians and other road users."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        
        result = predict_single_text(model, tokenizer, test_case['text'], classes, config.threshold)
        
        print(f"Text: {result['text_preview']}")
        print(f"Predicted IPC Sections: {result['predicted_classes']}")
        
        # Show top 5 probabilities
        probs = result['probabilities']
        top_indices = np.argsort(probs)[-5:][::-1]
        print("Top 5 highest probabilities:")
        for idx in top_indices:
            class_name = classes[idx]
            prob = probs[idx]
            print(f"  {class_name}: {prob:.3f}")

def main():
    """Main function to test and predict using the custom trained BERT model"""
    print("Custom Trained BERT Model Testing and Prediction")
    print("="*50)
    
    # Verify model files first
    if not verify_model_files():
        print("❌ Model verification failed. Please check the model files.")
        return
    
    # Display model metrics
    metrics = display_model_metrics()
    if metrics is None:
        print("❌ Could not load model metrics.")
        return
    
    # Load model and components
    model, tokenizer, classes = load_trained_model()
    
    # Predict on custom test cases
    predict_custom_test_cases(model, tokenizer, classes)
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
    print("Model prediction completed successfully!")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Number of IPC sections: {len(classes)}")
    print(f"Model performance: {metrics['f1']:.1%} F1 score")

if __name__ == "__main__":
    main() 