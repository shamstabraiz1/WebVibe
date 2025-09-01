"""Training script for Urdu sentiment analysis model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from src.models.urdu_sentiment_model import UrduSentimentModel
from src.data.text_preprocessor import PreprocessingConfig
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UrduSentimentDataset(Dataset):
    """Dataset class for Urdu sentiment data."""
    
    def __init__(self, texts, labels, model, max_length=512):
        self.texts = texts
        self.labels = labels
        self.model = model
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Preprocess and tokenize
        preprocessing_result = self.model.preprocessor.preprocess(text)
        processed_text = preprocessing_result.processed_text
        
        encoded = self.model.bert_model.tokenize_text(
            processed_text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_sample_data():
    """Load or create sample training data."""
    # This is a placeholder - replace with actual data loading
    sample_data = [
        ("یہ فلم بہت اچھی ہے", 4),  # Extremely Positive
        ("مجھے یہ پسند نہیں آیا", 0),  # Extremely Negative
        ("ٹھیک ہے، کوئی خاص بات نہیں", 2),  # Neutral
        ("واہ! کیا بات ہے", 3),  # Positive
        ("بہت برا تجربہ تھا", 1),  # Negative
        ("عام سی بات ہے", 2),  # Neutral
        ("سب سے بہترین فلم ہے", 4),  # Extremely Positive
        ("یہ بکواس ہے", 0),  # Extremely Negative
        ("کافی اچھی ہے", 3),  # Positive
        ("برا تجربہ تھا", 1),  # Negative
        # Add more sample data...
    ]
    
    # Duplicate data for demonstration
    sample_data = sample_data * 50  # 300 samples
    
    texts, labels = zip(*sample_data)
    return list(texts), list(labels)

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=3,
    learning_rate=2e-5,
    save_path=None
):
    """Train the sentiment analysis model."""
    
    device = model.device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    best_val_accuracy = 0
    training_history = []
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model.train_step(input_ids, attention_mask, labels, criterion)
            loss = outputs['loss']
            
            # Backward pass
            torch.tensor(loss, requires_grad=True).backward()
            optimizer.step()
            
            train_loss += loss
            train_correct += outputs['accuracy'] * len(labels)
            train_total += len(labels)
            
            train_pbar.set_postfix({
                'loss': f"{loss:.4f}",
                'acc': f"{outputs['accuracy']:.4f}"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model.evaluate_step(input_ids, attention_mask, labels, criterion)
                
                val_loss += outputs['loss']
                val_correct += outputs['accuracy'] * len(labels)
                val_total += len(labels)
                
                all_predictions.extend(outputs['predictions'])
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'loss': f"{outputs['loss']:.4f}",
                    'acc': f"{outputs['accuracy']:.4f}"
                })
        
        # Calculate metrics
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_accuracy': train_accuracy,
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }
        
        training_history.append(epoch_metrics)
        
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Train Loss: {epoch_metrics['train_loss']:.4f}, Train Acc: {train_accuracy:.4f}")
        logger.info(f"  Val Loss: {epoch_metrics['val_loss']:.4f}, Val Acc: {val_accuracy:.4f}")
        logger.info(f"  Val F1: {f1:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if save_path:
                model.save_model(save_path)
                logger.info(f"Best model saved with validation accuracy: {val_accuracy:.4f}")
        
        scheduler.step()
    
    return training_history

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Urdu Sentiment Analysis Model")
    parser.add_argument("--data_path", type=str, help="Path to training data CSV")
    parser.add_argument("--model_name", type=str, default=Config.MODEL_NAME, help="Pre-trained model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/urdu_sentiment_model", help="Model save path")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    
    args = parser.parse_args()
    
    # Load data
    if args.data_path and Path(args.data_path).exists():
        logger.info(f"Loading data from {args.data_path}")
        df = pd.read_csv(args.data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
    else:
        logger.info("Using sample data for demonstration")
        texts, labels = load_sample_data()
    
    logger.info(f"Loaded {len(texts)} samples")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    
    logger.info(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = UrduSentimentModel(
        model_name=args.model_name,
        num_classes=5,
        max_length=Config.MAX_LENGTH,
        preprocessing_config=PreprocessingConfig.standard()
    )
    
    # Create datasets
    train_dataset = UrduSentimentDataset(train_texts, train_labels, model)
    val_dataset = UrduSentimentDataset(val_texts, val_labels, model)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train model
    logger.info("Starting training...")
    training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.save_path
    )
    
    # Save training history
    history_path = Path(args.save_path).parent / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: {args.save_path}")
    logger.info(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    main()