"""Sentiment classification head for Urdu BERT model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import logging
import nltk
import spacy

logger = logging.getLogger(__name__)

class SentimentClassificationHead(nn.Module):
    """
    Classification head for sentiment analysis with confidence calibration.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 3,
        dropout_rate: float = 0.1,
        hidden_layers: Optional[list] = None,
        activation: str = 'relu',
        use_batch_norm: bool = False,
        temperature_scaling: bool = True
    ):
        """
        Initialize the classification head.
        
        Args:
            hidden_size: Size of input features from BERT
            num_classes: Number of sentiment classes
            dropout_rate: Dropout rate for regularization
            hidden_layers: List of hidden layer sizes (None for single layer)
            activation: Activation function ('relu', 'gelu', 'tanh')
            use_batch_norm: Whether to use batch normalization
            temperature_scaling: Whether to use temperature scaling for calibration
        """
        super(SentimentClassificationHead, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes or Config.NUM_CLASSES
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.temperature_scaling = temperature_scaling
        
        # Build the network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        current_size = hidden_size
        
        # Hidden layers (if specified)
        if hidden_layers:
            for hidden_size_layer in hidden_layers:
                self.layers.append(nn.Linear(current_size, hidden_size_layer))
                
                if use_batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_size_layer))
                
                # Activation function
                if activation.lower() == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation.lower() == 'gelu':
                    self.layers.append(nn.GELU())
                elif activation.lower() == 'tanh':
                    self.layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                
                self.layers.append(nn.Dropout(dropout_rate))
                current_size = hidden_size_layer
        
        # Output layer
        self.classifier = nn.Linear(current_size, num_classes)
        
        # Dropout for the final layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Temperature parameter for calibration
        if temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.ones(1))
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized SentimentClassificationHead with {self.count_parameters():,} parameters")
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_logits: bool = False,
        apply_temperature: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            hidden_states: Input features from BERT [batch_size, hidden_size]
            return_logits: Whether to return raw logits or probabilities
            apply_temperature: Whether to apply temperature scaling
            
        Returns:
            Classification outputs (logits or probabilities)
        """
        x = hidden_states
        
        # Apply dropout to input
        x = self.dropout(x)
        
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Final classification layer
        logits = self.classifier(x)
        
        if return_logits:
            return logits
        
        # Apply temperature scaling for calibration
        if apply_temperature and self.temperature_scaling:
            logits = logits / self.temperature
        
        # Return probabilities
        probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def predict(
        self,
        hidden_states: torch.Tensor,
        return_confidence: bool = True,
        confidence_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            hidden_states: Input features from BERT
            return_confidence: Whether to return confidence scores
            confidence_threshold: Threshold for high-confidence predictions
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            # Get probabilities
            probabilities = self.forward(hidden_states, return_logits=False)
            
            # Get predicted classes
            predicted_classes = torch.argmax(probabilities, dim=-1)
            
            # Get confidence scores (max probability)
            confidence_scores = torch.max(probabilities, dim=-1)[0]
            
            result = {
                'predictions': predicted_classes,
                'probabilities': probabilities
            }
            
            if return_confidence:
                result['confidence'] = confidence_scores
                result['high_confidence'] = confidence_scores > confidence_threshold
            
            return result
    
    def predict_with_uncertainty(
        self,
        hidden_states: torch.Tensor,
        num_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using Monte Carlo dropout.
        
        Args:
            hidden_states: Input features from BERT
            num_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            Dictionary containing predictions, confidence, and uncertainty
        """
        self.train()  # Enable dropout for uncertainty estimation
        
        predictions = []
        probabilities_list = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                probs = self.forward(hidden_states, return_logits=False)
                probabilities_list.append(probs)
                predictions.append(torch.argmax(probs, dim=-1))
        
        # Stack predictions and probabilities
        all_probabilities = torch.stack(probabilities_list)  # [num_samples, batch_size, num_classes]
        all_predictions = torch.stack(predictions)  # [num_samples, batch_size]
        
        # Calculate mean and variance
        mean_probabilities = torch.mean(all_probabilities, dim=0)
        var_probabilities = torch.var(all_probabilities, dim=0)
        
        # Predictive entropy (uncertainty measure)
        entropy = -torch.sum(mean_probabilities * torch.log(mean_probabilities + 1e-8), dim=-1)
        
        # Most frequent prediction
        mode_predictions = torch.mode(all_predictions, dim=0)[0]
        
        # Confidence as max mean probability
        confidence = torch.max(mean_probabilities, dim=-1)[0]
        
        return {
            'predictions': mode_predictions,
            'probabilities': mean_probabilities,
            'confidence': confidence,
            'uncertainty': entropy,
            'variance': var_probabilities
        }
    
    def calibrate_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Calibrate temperature parameter using validation data.
        
        Args:
            logits: Model logits on validation data
            labels: True labels for validation data
            lr: Learning rate for temperature optimization
            max_iter: Maximum number of optimization iterations
        """
        if not self.temperature_scaling:
            logger.warning("Temperature scaling is disabled")
            return
        
        # Create optimizer for temperature parameter only
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            # Apply current temperature
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        # Optimize temperature
        optimizer.step(eval_loss)
        
        logger.info(f"Temperature calibrated to: {self.temperature.item():.4f}")
    
    def get_feature_importance(
        self,
        hidden_states: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get feature importance using gradients.
        
        Args:
            hidden_states: Input features from BERT
            target_class: Target class for gradient computation (None for predicted class)
            
        Returns:
            Feature importance scores
        """
        hidden_states.requires_grad_(True)
        
        # Forward pass
        logits = self.forward(hidden_states, return_logits=True)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=-1)
        
        # Backward pass
        if logits.dim() > 1:
            # Batch processing
            target_logits = logits.gather(1, target_class.unsqueeze(1)).squeeze(1)
            grad_outputs = torch.ones_like(target_logits)
        else:
            target_logits = logits[target_class]
            grad_outputs = torch.ones_like(target_logits)
        
        gradients = torch.autograd.grad(
            outputs=target_logits,
            inputs=hidden_states,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Feature importance as absolute gradient values
        importance = torch.abs(gradients)
        
        return importance
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self) -> Dict[str, any]:
        """Get information about the classification head layers."""
        layer_info = []
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                layer_info.append({
                    'layer_type': 'Linear',
                    'input_size': layer.in_features,
                    'output_size': layer.out_features,
                    'parameters': layer.in_features * layer.out_features + layer.out_features
                })
            elif isinstance(layer, nn.BatchNorm1d):
                layer_info.append({
                    'layer_type': 'BatchNorm1d',
                    'features': layer.num_features,
                    'parameters': layer.num_features * 2
                })
            elif isinstance(layer, (nn.ReLU, nn.GELU, nn.Tanh)):
                layer_info.append({
                    'layer_type': layer.__class__.__name__,
                    'parameters': 0
                })
            elif isinstance(layer, nn.Dropout):
                layer_info.append({
                    'layer_type': 'Dropout',
                    'dropout_rate': layer.p,
                    'parameters': 0
                })
        
        # Add classifier layer
        layer_info.append({
            'layer_type': 'Classifier (Linear)',
            'input_size': self.classifier.in_features,
            'output_size': self.classifier.out_features,
            'parameters': self.classifier.in_features * self.classifier.out_features + self.classifier.out_features
        })
        
        return {
            'layers': layer_info,
            'total_parameters': self.count_parameters(),
            'temperature_scaling': self.temperature_scaling,
            'current_temperature': self.temperature.item() if self.temperature_scaling else 1.0
        }
    
    def reset_parameters(self):
        """Reset all parameters to initial values."""
        self._init_weights()
        if self.temperature_scaling:
            nn.init.constant_(self.temperature, 1.0)
        logger.info("Classification head parameters reset")

class MultiTaskClassificationHead(SentimentClassificationHead):
    """
    Multi-task classification head for sentiment and additional tasks.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        sentiment_classes: int = 3,
        dialect_classes: int = 6,
        formality_classes: int = 2,
        **kwargs
    ):
        """
        Initialize multi-task classification head.
        
        Args:
            hidden_size: Size of input features
            sentiment_classes: Number of sentiment classes
            dialect_classes: Number of dialect classes
            formality_classes: Number of formality classes (formal/informal)
            **kwargs: Additional arguments for parent class
        """
        # Initialize parent with sentiment classes
        super().__init__(hidden_size, sentiment_classes or Config.NUM_CLASSES, **kwargs)
        
        # Additional task heads
        self.dialect_classifier = nn.Linear(hidden_size, dialect_classes)
        self.formality_classifier = nn.Linear(hidden_size, formality_classes)
        
        # Initialize new layers
        nn.init.xavier_uniform_(self.dialect_classifier.weight)
        nn.init.xavier_uniform_(self.formality_classifier.weight)
        nn.init.constant_(self.dialect_classifier.bias, 0)
        nn.init.constant_(self.formality_classifier.bias, 0)
        
        logger.info("Initialized MultiTaskClassificationHead")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task: str = 'sentiment',
        return_logits: bool = False,
        apply_temperature: bool = True
    ) -> torch.Tensor:
        """
        Forward pass for specific task.
        
        Args:
            hidden_states: Input features from BERT
            task: Task to perform ('sentiment', 'dialect', 'formality', 'all')
            return_logits: Whether to return raw logits
            apply_temperature: Whether to apply temperature scaling
            
        Returns:
            Task-specific outputs
        """
        if task == 'sentiment':
            return super().forward(hidden_states, return_logits, apply_temperature)
        elif task == 'dialect':
            logits = self.dialect_classifier(self.dropout(hidden_states))
            return logits if return_logits else F.softmax(logits, dim=-1)
        elif task == 'formality':
            logits = self.formality_classifier(self.dropout(hidden_states))
            return logits if return_logits else F.softmax(logits, dim=-1)
        elif task == 'all':
            sentiment_out = super().forward(hidden_states, return_logits, apply_temperature)
            dialect_out = self.forward(hidden_states, 'dialect', return_logits, False)
            formality_out = self.forward(hidden_states, 'formality', return_logits, False)
            
            return {
                'sentiment': sentiment_out,
                'dialect': dialect_out,
                'formality': formality_out
            }
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def count_parameters(self) -> int:
        """Count parameters including additional task heads."""
        base_params = super().count_parameters()
        dialect_params = sum(p.numel() for p in self.dialect_classifier.parameters())
        formality_params = sum(p.numel() for p in self.formality_classifier.parameters())
        
        return base_params + dialect_params + formality_params

def main():
    # Download required NLTK data
    nltk.download('punkt')
    
    # Test NLTK
    text = "Hello! This is a test sentence for NLP processing."
    tokens = nltk.word_tokenize(text)
    print("NLTK tokenization:", tokens)
    
    # Test spaCy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    print("\nspaCy tokenization:", [token.text for token in doc])

if __name__ == "__main__":
    main()

# Download spaCy model
# python -m spacy download en_core_web_sm

# Run the script
# python src/main.py