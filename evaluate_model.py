"""Evaluation script for Urdu sentiment model."""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from src.models.urdu_sentiment_model import UrduSentimentModel

def evaluate_model():
    """Evaluate model performance."""
    # Load test data
    df = pd.read_csv("data/sample_data.csv")
    
    # Initialize model
    model = UrduSentimentModel()
    
    # Make predictions
    predictions = []
    true_labels = []
    
    for _, row in df.iterrows():
        result = model.predict(row['text'])
        
        # Convert sentiment to numeric
        sentiment_map = {
            'extremely_negative': 0, 'negative': 1, 'neutral': 2,
            'positive': 3, 'extremely_positive': 4
        }
        pred_label = sentiment_map[result['sentiment']]
        
        predictions.append(pred_label)
        true_labels.append(row['label'])
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, report

if __name__ == "__main__":
    evaluate_model()