"""Base BERT model wrapper for Urdu sentiment analysis."""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertModel,
    XLMRobertaTokenizer, XLMRobertaModel
)
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from config.config import Config

logger = logging.getLogger(__name__)

class UrduBertModel(nn.Module):
    """
    Base BERT model wrapper for Urdu text processing.
    Supports multiple pre-trained models including BERT and XLM-RoBERTa.
    """
    
    def __init__(
        self,
        model_name: str = None,
        num_classes: int = 3,
        max_length: int = 512,
        dropout_rate: float = 0.1,
        freeze_bert: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Urdu BERT model.
        
        Args:
            model_name: Name of the pre-trained model
            num_classes: Number of output classes
            max_length: Maximum sequence length
            dropout_rate: Dropout rate for classification head
            freeze_bert: Whether to freeze BERT parameters
            device: Device to load model on
        """
        super(UrduBertModel, self).__init__()
        
        self.model_name = model_name or Config.MODEL_NAME
        self.num_classes = num_classes or Config.NUM_CLASSES
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.freeze_bert = freeze_bert
        self.device = device or Config.get_device()
        
        # Initialize tokenizer and model
        self._load_model_and_tokenizer()
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized UrduBertModel with {self.model_name}")
        logger.info(f"Model parameters: {self.count_parameters():,}")
    
    def _load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                do_lower_case=False,  # Important for Urdu
                use_fast=True
            )
            
            # Load model configuration
            config = AutoConfig.from_pretrained(self.model_name)
            
            # Load the base model
            self.bert = AutoModel.from_pretrained(
                self.model_name,
                config=config
            )
            
            # Get hidden size from config
            self.hidden_size = config.hidden_size
            
            # Freeze BERT parameters if requested
            if self.freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
                logger.info("BERT parameters frozen")
            
            logger.info(f"Successfully loaded {self.model_name}")
            logger.info(f"Hidden size: {self.hidden_size}")
            logger.info(f"Vocabulary size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            # Fallback to alternative model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback model if primary model fails."""
        fallback_model = Config.ALTERNATIVE_MODEL
        logger.warning(f"Loading fallback model: {fallback_model}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                do_lower_case=False,
                use_fast=True
            )
            
            config = AutoConfig.from_pretrained(fallback_model)
            self.bert = AutoModel.from_pretrained(fallback_model, config=config)
            self.hidden_size = config.hidden_size
            self.model_name = fallback_model
            
            if self.freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            
            logger.info(f"Successfully loaded fallback model: {fallback_model}")
            
        except Exception as e:
            logger.error(f"Error loading fallback model: {e}")
            raise RuntimeError("Failed to load both primary and fallback models")
    
    def tokenize_text(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text.
        
        Args:
            text: Input text or list of texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Format of returned tensors
            
        Returns:
            Dictionary containing input_ids, attention_mask, etc.
        """
        try:
            encoded = self.tokenizer(
                text,
                padding=padding,
                truncation=truncation,
                max_length=self.max_length,
                return_tensors=return_tensors,
                return_attention_mask=True,
                return_token_type_ids=False  # Not needed for most models
            )
            
            # Move to device
            if return_tensors == "pt":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            return encoded
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode text using BERT.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            return_dict: Whether to return dictionary
            
        Returns:
            BERT outputs (last hidden state, pooler output, etc.)
        """
        try:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict
            )
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            BERT outputs (to be used by classification head)
        """
        outputs = self.encode_text(input_ids, attention_mask)
        
        # Return the pooled output (CLS token representation)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            # If no pooler output, use CLS token from last hidden state
            return outputs.last_hidden_state[:, 0, :]  # CLS token
    
    def get_embeddings(
        self,
        text: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get embeddings for input text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Text embeddings
        """
        self.eval()
        with torch.no_grad():
            encoded = self.tokenize_text(text)
            embeddings = self.forward(**encoded)
            return embeddings
    
    def save_model(self, save_path: Union[str, Path]):
        """
        Save the model and tokenizer.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the BERT model and tokenizer
            self.bert.save_pretrained(save_path / "bert")
            self.tokenizer.save_pretrained(save_path / "tokenizer")
            
            # Save model configuration
            model_config = {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'max_length': self.max_length,
                'dropout_rate': self.dropout_rate,
                'freeze_bert': self.freeze_bert,
                'hidden_size': self.hidden_size
            }
            
            torch.save(model_config, save_path / "model_config.pt")
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load_model(
        cls,
        load_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> 'UrduBertModel':
        """
        Load a saved model.
        
        Args:
            load_path: Path to load the model from
            device: Device to load model on
            
        Returns:
            Loaded UrduBertModel instance
        """
        load_path = Path(load_path)
        
        try:
            # Load model configuration
            model_config = torch.load(load_path / "model_config.pt", map_location='cpu')
            
            # Create model instance
            model = cls(
                model_name=model_config['model_name'],
                num_classes=model_config['num_classes'],
                max_length=model_config['max_length'],
                dropout_rate=model_config['dropout_rate'],
                freeze_bert=model_config['freeze_bert'],
                device=device
            )
            
            logger.info(f"Model loaded from {load_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'dropout_rate': self.dropout_rate,
            'freeze_bert': self.freeze_bert,
            'hidden_size': self.hidden_size,
            'vocab_size': len(self.tokenizer),
            'trainable_parameters': self.count_parameters(),
            'device': str(self.device)
        }
    
    def resize_token_embeddings(self, new_vocab_size: int):
        """
        Resize token embeddings (useful when adding new tokens).
        
        Args:
            new_vocab_size: New vocabulary size
        """
        self.bert.resize_token_embeddings(new_vocab_size)
        logger.info(f"Resized token embeddings to {new_vocab_size}")
    
    def add_special_tokens(self, special_tokens: Dict[str, str]):
        """
        Add special tokens to the tokenizer and resize embeddings.
        
        Args:
            special_tokens: Dictionary of special tokens to add
        """
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added} special tokens")
    
    def get_attention_weights(
        self,
        text: str,
        layer: int = -1
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Get attention weights for visualization.
        
        Args:
            text: Input text
            layer: Which layer's attention to return (-1 for last layer)
            
        Returns:
            Attention weights and tokens
        """
        self.eval()
        with torch.no_grad():
            encoded = self.tokenize_text(text)
            
            # Get outputs with attention weights
            outputs = self.bert(
                **encoded,
                output_attentions=True,
                return_dict=True
            )
            
            # Get attention weights from specified layer
            attention_weights = outputs.attentions[layer]
            
            # Get tokens for visualization
            tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            
            return attention_weights[0], tokens  # Return first batch item