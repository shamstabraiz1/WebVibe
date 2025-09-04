"""Complete Urdu sentiment analysis model integrating all components."""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path

from .urdu_bert_model import UrduBertModel
from .classification_head import SentimentClassificationHead
from .data_models import SentimentLabel, DialectHint
from ..data.text_preprocessor import TextPreprocessor, PreprocessingConfig

logger = logging.getLogger(__name__)

class UrduSentimentModel(nn.Module):
    """
    Complete Urdu sentiment analysis model with preprocessing and classification.
    """
    
    def __init__(
        self,
    model_name: str = "distilbert-base-multilingual-cased",
        num_classes: int = 3,
        max_length: int = 512,
        dropout_rate: float = 0.1,
        hidden_layers: Optional[List[int]] = None,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the complete Urdu sentiment model.
        
        Args:
            model_name: Pre-trained model name
            num_classes: Number of sentiment classes
            max_length: Maximum sequence length
            dropout_rate: Dropout rate for classification
            hidden_layers: Hidden layer sizes for classification head
            preprocessing_config: Text preprocessing configuration
            device: Device to run model on
        """
        super(UrduSentimentModel, self).__init__()
        
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.bert_model = UrduBertModel(
            model_name=model_name,
            num_classes=num_classes,
            max_length=max_length,
            device=self.device
        )
        
        self.classification_head = SentimentClassificationHead(
            hidden_size=self.bert_model.hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            hidden_layers=hidden_layers
        )

        # Ensure classification head matches model dtype if quantized
        if hasattr(self.bert_model.bert, 'dtype') and self.bert_model.bert.dtype == torch.float16:
            self.classification_head = self.classification_head.half()
        
        # Text preprocessor
        self.preprocessor = TextPreprocessor(
            preprocessing_config or PreprocessingConfig.standard()
        )
        
        # Label mappings
        self.id_to_label = {
            0: SentimentLabel.EXTREMELY_NEGATIVE,
            1: SentimentLabel.NEGATIVE,
            2: SentimentLabel.NEUTRAL,
            3: SentimentLabel.POSITIVE,
            4: SentimentLabel.EXTREMELY_POSITIVE
        }
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized UrduSentimentModel on {self.device}")
        logger.info(f"Total parameters: {self.count_parameters():,}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            return_logits: Whether to return logits or probabilities
            
        Returns:
            Model outputs (logits or probabilities)
        """
        # Get BERT embeddings
        bert_outputs = self.bert_model(input_ids, attention_mask)
        
        # Classification
        outputs = self.classification_head(bert_outputs, return_logits=return_logits)
        
        return outputs
    
    def predict(
        self,
        text: str,
        return_confidence: bool = True,
        return_processing_time: bool = True,
        preprocess: bool = True
    ) -> Dict[str, any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            return_confidence: Whether to return confidence scores
            return_processing_time: Whether to return processing time
            preprocess: Whether to preprocess the text
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        self.eval()
        with torch.no_grad():
            # Preprocess text if requested
            processed_text = text
            preprocessing_info = None
            
            if preprocess:
                preprocessing_result = self.preprocessor.preprocess(text)
                processed_text = preprocessing_result.processed_text
                preprocessing_info = {
                    'original_script_type': preprocessing_result.original_script_type,
                    'detected_dialect': preprocessing_result.detected_dialect.value,
                    'transformations_applied': preprocessing_result.transformations_applied
                }
            
            # Tokenize
            encoded = self.bert_model.tokenize_text(processed_text)
            
            # Forward pass
            probabilities = self.forward(**encoded, return_logits=False)
            
            # Get prediction
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            predicted_label = self.id_to_label[predicted_class_id]
            
            # Get confidence
            confidence = torch.max(probabilities, dim=-1)[0].item()
            
            # Prepare result
            result = {
                'text': text,
                'processed_text': processed_text,
                'sentiment': predicted_label.value,
                'confidence': confidence,
                'probabilities': {
                    'extremely_negative': probabilities[0][0].item(),
                    'negative': probabilities[0][1].item(),
                    'neutral': probabilities[0][2].item(),
                    'positive': probabilities[0][3].item(),
                    'extremely_positive': probabilities[0][4].item()
                }
            }
            
            if preprocessing_info:
                result['preprocessing_info'] = preprocessing_info
            
            if return_processing_time:
                result['processing_time'] = time.time() - start_time
            
            return result
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        return_confidence: bool = True,
        return_processing_time: bool = True,
        preprocess: bool = True
    ) -> List[Dict[str, any]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            return_confidence: Whether to return confidence scores
            return_processing_time: Whether to return processing time
            preprocess: Whether to preprocess texts
            
        Returns:
            List of prediction results
        """
        start_time = time.time()
        results = []
        
        self.eval()
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Preprocess batch if requested
                processed_texts = batch_texts
                preprocessing_infos = []
                
                if preprocess:
                    preprocessing_results = self.preprocessor.preprocess_batch(batch_texts)
                    processed_texts = [r.processed_text for r in preprocessing_results]
                    preprocessing_infos = [
                        {
                            'original_script_type': r.original_script_type,
                            'detected_dialect': r.detected_dialect.value,
                            'transformations_applied': r.transformations_applied
                        }
                        for r in preprocessing_results
                    ]
                
                # Tokenize batch
                encoded = self.bert_model.tokenize_text(processed_texts)
                
                # Forward pass
                probabilities = self.forward(**encoded, return_logits=False)
                
                # Process results
                for j, (original_text, processed_text) in enumerate(zip(batch_texts, processed_texts)):
                    probs = probabilities[j]
                    predicted_class_id = torch.argmax(probs).item()
                    predicted_label = self.id_to_label[predicted_class_id]
                    confidence = torch.max(probs).item()
                    
                    result = {
                        'text': original_text,
                        'processed_text': processed_text,
                        'sentiment': predicted_label.value,
                        'confidence': confidence,
                        'probabilities': {
                            'extremely_negative': probs[0].item(),
                            'negative': probs[1].item(),
                            'neutral': probs[2].item(),
                            'positive': probs[3].item(),
                            'extremely_positive': probs[4].item()
                        }
                    }
                    
                    if preprocessing_infos:
                        result['preprocessing_info'] = preprocessing_infos[j]
                    
                    results.append(result)
        
        if return_processing_time:
            total_time = time.time() - start_time
            for result in results:
                result['processing_time'] = total_time / len(results)
        
        return results
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: True labels
            criterion: Loss function
            
        Returns:
            Training metrics
        """
        self.train()
        
        # Forward pass
        logits = self.forward(input_ids, attention_mask, return_logits=True)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def evaluate_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Single evaluation step.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: True labels
            criterion: Loss function
            
        Returns:
            Evaluation metrics
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            logits = self.forward(input_ids, attention_mask, return_logits=True)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean()
            
            # Get probabilities for confidence analysis
            probabilities = torch.softmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0].mean()
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'confidence': confidence.item(),
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy()
            }
    
    def save_model(self, save_path: Union[str, Path]):
        """
        Save the complete model.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save BERT model
        self.bert_model.save_model(save_path / "bert_model")
        
        # Save classification head
        torch.save(self.classification_head.state_dict(), save_path / "classification_head.pt")
        
        # Save preprocessor config
        torch.save(self.preprocessor.config, save_path / "preprocessing_config.pt")
        
        # Save model metadata
        metadata = {
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'id_to_label': self.id_to_label,
            'label_to_id': self.label_to_id
        }
        torch.save(metadata, save_path / "model_metadata.pt")
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(
        cls,
        load_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> 'UrduSentimentModel':
        """
        Load a saved model.
        
        Args:
            load_path: Path to load the model from
            device: Device to load model on
            
        Returns:
            Loaded UrduSentimentModel instance
        """
        load_path = Path(load_path)
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load metadata
        metadata = torch.load(load_path / "model_metadata.pt", map_location='cpu')
        
        # Load preprocessing config
        preprocessing_config = torch.load(load_path / "preprocessing_config.pt", map_location='cpu')
        
        # Load BERT model
        bert_model = UrduBertModel.load_model(load_path / "bert_model", device)
        
        # Create model instance
        model = cls(
            model_name=bert_model.model_name,
            num_classes=metadata['num_classes'],
            max_length=metadata['max_length'],
            preprocessing_config=preprocessing_config,
            device=device
        )
        
        # Load classification head weights
        classification_head_state = torch.load(
            load_path / "classification_head.pt", 
            map_location=device
        )
        model.classification_head.load_state_dict(classification_head_state)
        
        # Restore label mappings
        model.id_to_label = metadata['id_to_label']
        model.label_to_id = metadata['label_to_id']
        
        logger.info(f"Model loaded from {load_path}")
        return model
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get comprehensive model information."""
        return {
            'model_name': self.bert_model.model_name,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'device': str(self.device),
            'total_parameters': self.count_parameters(),
            'bert_parameters': self.bert_model.count_parameters(),
            'classification_parameters': self.classification_head.count_parameters(),
            'preprocessing_config': self.preprocessor.config.__dict__,
            'label_mappings': {k: v.value for k, v in self.id_to_label.items()}
        }
    
    def set_preprocessing_config(self, config: PreprocessingConfig):
        """Update preprocessing configuration."""
        self.preprocessor.update_config(config)
        logger.info("Preprocessing configuration updated")
    
    def freeze_bert(self):
        """Freeze BERT parameters for fine-tuning only the classification head."""
        for param in self.bert_model.parameters():
            param.requires_grad = False
        logger.info("BERT parameters frozen")
    
    def unfreeze_bert(self):
        """Unfreeze BERT parameters for full model training."""
        for param in self.bert_model.parameters():
            param.requires_grad = True
        logger.info("BERT parameters unfrozen")
    
    def get_embeddings(self, text: str, preprocess: bool = True) -> torch.Tensor:
        """
        Get text embeddings from BERT.
        
        Args:
            text: Input text
            preprocess: Whether to preprocess text
            
        Returns:
            Text embeddings
        """
        processed_text = text
        if preprocess:
            preprocessing_result = self.preprocessor.preprocess(text)
            processed_text = preprocessing_result.processed_text
        
        return self.bert_model.get_embeddings(processed_text)