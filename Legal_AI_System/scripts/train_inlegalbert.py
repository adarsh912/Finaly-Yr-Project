import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Try importing colab drive only if running in Google Colab environment
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

from pathlib import Path

# Define paths for model, dataset, and output
if IN_COLAB:
    MODEL_PATH = "/content/drive/MyDrive/Final Year Project/InLegalBERT"
    DATASET_PATH = "/content/drive/MyDrive/Final Year Project/ILSI/Dataset"
    OUTPUT_DIR = "/content/drive/MyDrive/Final Year Project/trained_model"
else:
    # Resolve relative paths dynamically
    base_dir = Path(__file__).resolve().parents[2]
    if (base_dir / "Dataset").exists():
        DATASET_PATH = str(base_dir / "Dataset")
    else:
        DATASET_PATH = "Dataset"
    MODEL_PATH = "trained_model"
    OUTPUT_DIR = "trained_model"

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda import amp
from torch.nn import BCEWithLogitsLoss

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

#######################
# Training Arguments #
#######################
class TrainingArgs:
    """
    Configuration class containing all the hyperparameters for training
    - Model parameters (max_length, batch sizes)
    - Training parameters (learning rate, epochs, warmup)
    - Optimization parameters (weight decay, gradient clipping)
    - Dataset size limits for reduced training time
    """
    def __init__(self):
        # Model parameters
        self.max_length = 512
        self.train_batch_size = 8
        self.eval_batch_size = 8

        # Training parameters
        self.learning_rate = 2e-5
        self.num_train_epochs = 5
        self.warmup_ratio = 0.1

        # Optimization parameters
        self.weight_decay = 0.01
        self.gradient_accumulation_steps = 4
        self.fp16 = True
        self.max_grad_norm = 1.0

        # Dataset size limits
        self.max_train_samples = 15000
        self.max_eval_samples = 3000
        self.max_test_samples = 5000

args = TrainingArgs()

#######################
# Memory Optimization #
#######################
def apply_memory_optimizations():
    """
    Apply various memory optimizations for efficient training:
    - Enable TF32 for better performance on NVIDIA GPUs
    - Enable cuDNN benchmarking for faster training
    - Enable automatic mixed precision
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

#######################
# Text Preprocessing #
#######################
def clean_legal_text(text):
    """
    Clean and standardize legal text:
    - Handle different input types (list, string)
    - Remove extra whitespace
    - Standardize section numbers
    - Remove citation numbers
    """
    # Handle list input
    if isinstance(text, list):
        text = ' '.join(text)
    elif not isinstance(text, str):
        text = str(text)

    # Remove extra whitespace
    text = ' '.join(text.split())
    # Standardize section numbers
    text = re.sub(r'Section\s+(\d+)', r'Section\1', text)
    # Standardize citations
    text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers
    return text

def truncate_with_overlap(text, max_length, overlap=50):
    """
    Handle long documents by truncating with overlap:
    - Keep documents within max_length
    - For long documents, create overlapping chunks
    - Return first chunk for training
    """
    if len(text) <= max_length:
        return text

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + max_length]
        chunks.append(chunk)
        start += (max_length - overlap)

    return chunks[0]  # For training, take first chunk

#######################
# Configuration Class #
#######################
class Config:
    """
    Global configuration class for model and training settings
    Initializes device and training parameters
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 512
        self.train_batch_size = 8
        self.eval_batch_size = 8
        self.learning_rate = 1e-5  # Further reduced for stability
        self.weight_decay = 0.01
        self.epochs = 6  # Increased epochs for better convergence
        self.max_grad_norm = 1.0
        self.warmup_ratio = 0.1
        self.gradient_accumulation_steps = 4
        self.pos_weight_multiplier = 12.0  # Increased for better class balance
        self.focal_gamma = 2.0  # Focal loss gamma parameter
        self.threshold = 0.25  # Lowered threshold for better recall
        self.dropout_rate = 0.15  # Increased dropout for better generalization

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

#######################
# Memory Monitoring #
#######################
def monitor_memory():
    """
    Monitor GPU memory usage during training
    Prints allocated and cached memory in GB
    """
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

#######################
# Data Loading #
#######################
def load_statutes():
    """
    Load the IPC statutes file and create a mapping of section IDs to their descriptions
    This helps the model better understand the legal context of each section
    """
    statutes_map = {}
    print("Loading statutes data...")
    with open(os.path.join(DATASET_PATH, 'statutes-00000-of-00001.json'), 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'id' in item and 'text' in item:
                section_id = item['id']
                description = ' '.join(item['text'])
                statutes_map[section_id] = description
    return statutes_map

def load_data():
    """
    Load and preprocess the dataset with memory efficiency:
    1. Load data from JSON files
    2. Validate data structure
    3. Apply random sampling for reduced dataset
    4. Clean and process texts
    5. Apply truncation with overlap
    6. Load statutes for reference
    7. Load test set for final evaluation
    """
    train_data = []
    valid_data = []
    test_data = []

    print("Loading training data...")
    # Load training data
    train_file = os.path.join(DATASET_PATH, 'train-00000-of-00001.json')
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'text' in item and 'labels' in item:  # Validate data structure
                    train_data.append(item)
    else:
        # Load from split part files
        part_idx = 1
        while True:
            part_file = os.path.join(DATASET_PATH, f'train-00000-of-00001_part{part_idx}.json')
            if not os.path.exists(part_file):
                break
            print(f"Loading train part {part_idx}...")
            with open(part_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if 'text' in item and 'labels' in item:
                        train_data.append(item)
            part_idx += 1

    print("Loading validation data...")
    # Load validation data
    dev_file = os.path.join(DATASET_PATH, 'dev-00000-of-00001.json')
    if os.path.exists(dev_file):
        with open(dev_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'text' in item and 'labels' in item:  # Validate data structure
                    valid_data.append(item)
    else:
        # Load from split part files
        part_idx = 1
        while True:
            part_file = os.path.join(DATASET_PATH, f'dev-00000-of-00001_part{part_idx}.json')
            if not os.path.exists(part_file):
                break
            print(f"Loading dev part {part_idx}...")
            with open(part_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if 'text' in item and 'labels' in item:
                        valid_data.append(item)
            part_idx += 1

    print("Loading test data...")
    # Load test data
    with open(os.path.join(DATASET_PATH, 'test-00000-of-00001.json'), 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'text' in item and 'labels' in item:  # Validate data structure
                    test_data.append(item)

    # Load statutes for reference
    statutes_map = load_statutes()
    print(f"Loaded {len(statutes_map)} statute descriptions")

    # Randomly sample the data
    if len(train_data) > args.max_train_samples:
        print(f"Reducing training data from {len(train_data)} to {args.max_train_samples} samples")
        train_data = np.random.choice(train_data, args.max_train_samples, replace=False).tolist()

    if len(valid_data) > args.max_eval_samples:
        print(f"Reducing validation data from {len(valid_data)} to {args.max_eval_samples} samples")
        valid_data = np.random.choice(valid_data, args.max_eval_samples, replace=False).tolist()

    if len(test_data) > args.max_test_samples:
        print(f"Reducing test data from {len(test_data)} to {args.max_test_samples} samples")
        test_data = np.random.choice(test_data, args.max_test_samples, replace=False).tolist()

    print(f"Processing {len(train_data)} training samples, {len(valid_data)} validation samples, and {len(test_data)} test samples...")

    # Extract texts and labels
    train_texts = [clean_legal_text(item['text']) for item in train_data]
    train_labels = [item['labels'] for item in train_data]
    valid_texts = [clean_legal_text(item['text']) for item in valid_data]
    valid_labels = [item['labels'] for item in valid_data]
    test_texts = [clean_legal_text(item['text']) for item in test_data]
    test_labels = [item['labels'] for item in test_data]

    print("After cleaning:", len(train_texts), "training samples and", len(valid_texts), "validation samples")

    return train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels, statutes_map

#######################
# Dataset Class #
#######################
class LegalDataset(Dataset):
    """
    Custom Dataset class for legal documents:
    - Handles text and label pairs
    - Performs tokenization
    - Converts labels to multi-hot encoding
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]  # Text is already cleaned and should be a string
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension added by tokenizer
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

        return item

#######################
# Training Functions #
#######################
def train_epoch(model, data_loader, optimizer, scheduler, device, scaler, class_weights):
    """
    Train for one epoch with improvements:
    - Class-specific weighted BCE Loss for handling class imbalance
    - Gradient clipping
    - Learning rate warmup
    - Better loss scaling
    """
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass with mixed precision
        with amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Get logits and compute weighted BCE loss
            logits = outputs.logits

            # Compute class-specific weighted BCE loss
            pos_weight = class_weights.to(device)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight,
                reduction='none'
            )

            # Add focal loss component for hard examples
            probs = torch.sigmoid(logits)
            pt = torch.where(labels == 1, probs, 1 - probs)
            focal_weight = (1 - pt) ** config.focal_gamma

            # Combine losses with class-specific weighting
            loss = (focal_weight * bce_loss).mean()

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step with scaling
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        # Clear memory
        del outputs, loss
        torch.cuda.empty_cache()

    return total_loss / len(data_loader)

#######################
# Evaluation Functions #
#######################
def evaluate_model(model, data_loader, device, class_weights):
    """
    Evaluate the model with improved metrics and thresholding
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            probs = torch.sigmoid(logits)

            # Use improved threshold for predictions
            predictions = (probs > config.threshold).float()

            # Compute loss using class-specific weights
            pos_weight = class_weights.to(device)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight,
                reduction='none'
            )
            focal_weight = (1 - probs) ** config.focal_gamma
            loss = (focal_weight * bce_loss).mean()

            total_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    h_loss = hamming_loss(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )

    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'hamming_loss': h_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics

def evaluate_test_set(model, test_loader, device, mlb, class_weights):
    """
    Evaluate the model on the test set and print detailed metrics
    This should only be called after training is complete
    """
    model.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            probs = torch.sigmoid(logits)

            # Use improved threshold for predictions
            predictions = (probs > config.threshold).float()

            # Compute loss using class-specific weights
            pos_weight = class_weights.to(device)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight,
                reduction='none'
            )
            focal_weight = (1 - probs) ** config.focal_gamma
            loss = (focal_weight * bce_loss).mean()

            test_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert predictions and labels to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    h_loss = hamming_loss(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )

    # Print per-class metrics
    class_names = mlb.classes_
    per_class_precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)

    print("\nTest Set Metrics:")
    print(f"Loss: {test_loss/len(test_loader):.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Hamming Loss: {h_loss:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")

    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"\nClass {class_name}:")
        print(f"Precision: {per_class_precision[i]:.4f}")
        print(f"Recall: {per_class_recall[i]:.4f}")
        print(f"F1: {per_class_f1[i]:.4f}")

    return {
        'loss': test_loss/len(test_loader),
        'accuracy': accuracy,
        'hamming_loss': h_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

#######################
# Model Saving #
#######################
def save_best_model(model, tokenizer, metrics, best_metric, output_dir):
    """
    Save model when it achieves better performance:
    - Compares current F1 score with best score
    - Saves both model and tokenizer
    - Returns updated best metric
    """
    # Note: best_metric is already set to the new best value in the caller,
    # so we save the model whenever this function is called.
    print(f"💾 Saving new best model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return metrics['f1']

#######################
# Class Weight Computation #
#######################
def compute_class_weights(label_counts, method='balanced'):
    """
    Compute class-specific weights for handling imbalanced data
    """
    if method == 'balanced':
        # Balanced weights: inverse of class frequency with clipping
        total_samples = label_counts.sum()
        class_weights = total_samples / (len(label_counts) * label_counts)
        # Clip weights to prevent extreme values
        class_weights = np.clip(class_weights, 0.1, 10.0)
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(label_counts)
    elif method == 'sqrt':
        # Square root of inverse frequency
        total_samples = label_counts.sum()
        class_weights = np.sqrt(total_samples / (len(label_counts) * label_counts))
        class_weights = np.clip(class_weights, 0.1, 5.0)
        class_weights = class_weights / class_weights.sum() * len(label_counts)
    else:
        # Equal weights
        class_weights = np.ones(len(label_counts))

    return torch.FloatTensor(class_weights)

#######################
# Main Training Loop #
#######################
def main():
    """
    Main training function:
    1. Sets up random seeds for reproducibility
    2. Applies memory optimizations
    3. Loads and preprocesses data
    4. Initializes model, optimizer, and scheduler
    5. Trains and evaluates the model
    6. Saves the best model
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Apply memory optimizations
    apply_memory_optimizations()

    # Load data including test set and statutes
    train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels, statutes_map = load_data()

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels + valid_labels + test_labels)
    num_labels = len(mlb.classes_)

    # Save classes mapping (required by predictors to decode predictions)
    classes_list = list(mlb.classes_)
    classes_path = os.path.join(OUTPUT_DIR, 'classes.json')
    with open(classes_path, 'w') as f:
        json.dump(classes_list, f, indent=2)
    print(f"✅ Saved {len(classes_list)} classes to {classes_path}")

    # Convert labels to multi-hot encoding
    train_labels_encoded = mlb.transform(train_labels)
    valid_labels_encoded = mlb.transform(valid_labels)
    test_labels_encoded = mlb.transform(test_labels)

    # Print label distribution
    print("\nLabel Distribution:")
    print(f"Number of unique labels: {num_labels}")
    label_counts = train_labels_encoded.sum(axis=0)
    for i, (label, count) in enumerate(zip(mlb.classes_, label_counts)):
        print(f"Label {i} ({label}): {count} occurrences ({count/len(train_labels_encoded)*100:.2f}%)")
        if label in statutes_map:
            print(f"Description: {statutes_map[label][:100]}...")

    # Initialize tokenizer and model with better configuration
    tokenizer = AutoTokenizer.from_pretrained('law-ai/InLegalBERT')
    model = AutoModelForSequenceClassification.from_pretrained(
        'law-ai/InLegalBERT',
        num_labels=num_labels,
        problem_type="multi_label_classification",
        hidden_dropout_prob=config.dropout_rate,  # Increased dropout
        attention_probs_dropout_prob=config.dropout_rate,  # Increased dropout
    )

    # Initialize classifier layer with better scaling
    if hasattr(model, 'classifier'):
        if hasattr(model.classifier, 'weight'):
            torch.nn.init.xavier_uniform_(model.classifier.weight, gain=1.0)
        if hasattr(model.classifier, 'bias'):
            torch.nn.init.zeros_(model.classifier.bias)

    # Compute class-specific weights for better handling of imbalanced data
    class_weights = compute_class_weights(label_counts, method='balanced')
    print(f"\nClass weights computed using balanced method")
    print(f"Min weight: {class_weights.min():.3f}, Max weight: {class_weights.max():.3f}")

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create datasets and dataloaders
    train_dataset = LegalDataset(train_texts, train_labels_encoded, tokenizer, config.max_length)
    valid_dataset = LegalDataset(valid_texts, valid_labels_encoded, tokenizer, config.max_length)
    test_dataset = LegalDataset(test_texts, test_labels_encoded, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=2
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=2
    )

    # Move model to device
    model = model.to(config.device)

    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )

    # Create learning rate scheduler with warmup and cosine annealing
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Initialize gradient scaler for mixed precision
    scaler = amp.GradScaler()

    # Training loop with improved monitoring and early stopping
    best_f1 = 0
    patience = 5  # Increased patience for longer training
    patience_counter = 0

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Threshold: {config.threshold}")
    print(f"Dropout rate: {config.dropout_rate}")
    print(f"Class weights method: balanced")

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        monitor_memory()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config.device, scaler, class_weights)

        # Evaluate
        monitor_memory()
        val_metrics = evaluate_model(model, valid_loader, config.device, class_weights)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print("Validation Metrics:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")

        # Debug: Check if model is predicting all zeros
        if val_metrics['precision'] == 0.0 and val_metrics['recall'] == 0.0:
            print("⚠️  WARNING: Model is predicting all zeros! This indicates a training issue.")
            print("   - Check if threshold is too high")
            print("   - Check if class weights are too extreme")
            print("   - Consider reducing learning rate further")

        # Save best model and early stopping based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_best_model(model, tokenizer, val_metrics, best_f1, OUTPUT_DIR)
            patience_counter = 0
            print(f"New best F1 score: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best F1 score achieved: {best_f1:.4f}")
                break

        # Clear memory
        torch.cuda.empty_cache()

    # Final evaluation on test set
    print("\nFinal evaluation on test set:")
    test_metrics = evaluate_test_set(model, test_loader, config.device, mlb, class_weights)

    # Save test metrics
    test_metrics_path = os.path.join(OUTPUT_DIR, 'test_metrics.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

if __name__ == "__main__":
    main()