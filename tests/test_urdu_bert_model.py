"""Tests for Urdu BERT model wrapper."""

import pytest
import torch
from unittest.mock import Mock, patch
from pathlib import Path

from src.models.urdu_bert_model import UrduBertModel

class TestUrduBertModel:
    """Test cases for UrduBertModel."""
    
    @pytest.fixture
    def mock_model_components(self):
        """Mock the transformers components to avoid downloading models."""
        with patch('src.models.urdu_bert_model.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.urdu_bert_model.AutoModel') as mock_model, \
             patch('src.models.urdu_bert_model.AutoConfig') as mock_config:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.__len__ = Mock(return_value=30000)
            mock_tokenizer_instance.convert_ids_to_tokens = Mock(return_value=['[CLS]', 'test', '[SEP]'])
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock config
            mock_config_instance = Mock()
            mock_config_instance.hidden_size = 768
            mock_config.from_pretrained.return_value = mock_config_instance
            
            # Mock model
            mock_model_instance = Mock()
            mock_model_instance.parameters.return_value = [torch.randn(100, 100)]
            mock_model.from_pretrained.return_value = mock_model_instance
            
            yield {
                'tokenizer': mock_tokenizer_instance,
                'model': mock_model_instance,
                'config': mock_config_instance
            }
    
    def test_model_initialization(self, mock_model_components):
        """Test model initialization."""
        model = UrduBertModel(
            model_name="bert-base-multilingual-cased",
            num_classes=5,
            max_length=512,
            device=torch.device('cpu')
        )
        
        assert model.model_name == "bert-base-multilingual-cased"
        assert model.num_classes == 5
        assert model.max_length == 512
        assert model.hidden_size == 768
    
    def test_model_initialization_with_defaults(self, mock_model_components):
        """Test model initialization with default parameters."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        assert model.num_classes == 5  # Default
        assert model.max_length == 512  # Default from config
        assert model.dropout_rate == 0.1  # Default
    
    def test_tokenize_text_single(self, mock_model_components):
        """Test tokenization of single text."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        # Mock tokenizer return
        mock_encoded = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        model.tokenizer.return_value = mock_encoded
        
        result = model.tokenize_text("یہ ٹیسٹ ہے")
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        model.tokenizer.assert_called_once()
    
    def test_tokenize_text_batch(self, mock_model_components):
        """Test tokenization of batch of texts."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        # Mock tokenizer return
        mock_encoded = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        model.tokenizer.return_value = mock_encoded
        
        texts = ["یہ ٹیسٹ ہے", "دوسرا ٹیسٹ"]
        result = model.tokenize_text(texts)
        
        assert result['input_ids'].shape[0] == 2  # Batch size
        model.tokenizer.assert_called_once_with(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False
        )
    
    def test_encode_text(self, mock_model_components):
        """Test text encoding through BERT."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        # Mock BERT output
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_output.pooler_output = torch.randn(1, 768)
        model.bert.return_value = mock_output
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        result = model.encode_text(input_ids, attention_mask)
        
        model.bert.assert_called_once_with(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        assert result == mock_output
    
    def test_forward_with_pooler_output(self, mock_model_components):
        """Test forward pass with pooler output."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        # Mock BERT output with pooler_output
        mock_output = Mock()
        mock_output.pooler_output = torch.randn(1, 768)
        model.bert.return_value = mock_output
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        result = model.forward(input_ids, attention_mask)
        
        assert result.shape == (1, 768)
        assert torch.equal(result, mock_output.pooler_output)
    
    def test_forward_without_pooler_output(self, mock_model_components):
        """Test forward pass without pooler output (use CLS token)."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        # Mock BERT output without pooler_output
        mock_output = Mock()
        mock_output.pooler_output = None
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        model.bert.return_value = mock_output
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        result = model.forward(input_ids, attention_mask)
        
        assert result.shape == (1, 768)
        # Should return CLS token (first token) from last hidden state
        expected = mock_output.last_hidden_state[:, 0, :]
        assert torch.equal(result, expected)
    
    def test_get_embeddings(self, mock_model_components):
        """Test getting embeddings for text."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        # Mock tokenizer and forward pass
        mock_encoded = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        model.tokenizer.return_value = mock_encoded
        
        mock_embeddings = torch.randn(1, 768)
        with patch.object(model, 'forward', return_value=mock_embeddings):
            result = model.get_embeddings("یہ ٹیسٹ ہے")
            
            assert torch.equal(result, mock_embeddings)
    
    def test_count_parameters(self, mock_model_components):
        """Test parameter counting."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        # Mock parameters
        param1 = torch.randn(100, 100)
        param1.requires_grad = True
        param2 = torch.randn(50, 50)
        param2.requires_grad = False
        param3 = torch.randn(10, 10)
        param3.requires_grad = True
        
        with patch.object(model, 'parameters', return_value=[param1, param2, param3]):
            count = model.count_parameters()
            
            # Should only count trainable parameters
            expected = param1.numel() + param3.numel()  # param2 is not trainable
            assert count == expected
    
    def test_get_model_info(self, mock_model_components):
        """Test getting model information."""
        model = UrduBertModel(
            model_name="test-model",
            num_classes=5,
            max_length=256,
            device=torch.device('cpu')
        )
        
        with patch.object(model, 'count_parameters', return_value=1000):
            info = model.get_model_info()
            
            assert info['model_name'] == "test-model"
            assert info['num_classes'] == 5
            assert info['max_length'] == 256
            assert info['trainable_parameters'] == 1000
            assert info['device'] == 'cpu'
    
    def test_freeze_bert_parameters(self, mock_model_components):
        """Test freezing BERT parameters."""
        # Mock parameters
        param1 = Mock()
        param2 = Mock()
        mock_model_components['model'].parameters.return_value = [param1, param2]
        
        model = UrduBertModel(freeze_bert=True, device=torch.device('cpu'))
        
        # Check that requires_grad was set to False
        param1.requires_grad = False
        param2.requires_grad = False
    
    def test_fallback_model_loading(self, mock_model_components):
        """Test fallback model loading when primary model fails."""
        with patch('src.models.urdu_bert_model.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.urdu_bert_model.AutoModel') as mock_model, \
             patch('src.models.urdu_bert_model.AutoConfig') as mock_config:
            
            # Make primary model fail
            mock_tokenizer.from_pretrained.side_effect = [Exception("Primary failed"), Mock()]
            mock_config.from_pretrained.side_effect = [Exception("Primary failed"), Mock(hidden_size=768)]
            mock_model.from_pretrained.side_effect = [Exception("Primary failed"), Mock()]
            
            # Should not raise exception due to fallback
            model = UrduBertModel(
                model_name="failing-model",
                device=torch.device('cpu')
            )
            
            # Should have loaded fallback model
            assert mock_tokenizer.from_pretrained.call_count == 2
    
    def test_add_special_tokens(self, mock_model_components):
        """Test adding special tokens."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        # Mock tokenizer methods
        model.tokenizer.add_special_tokens.return_value = 2
        model.tokenizer.__len__ = Mock(return_value=30002)
        
        with patch.object(model, 'resize_token_embeddings') as mock_resize:
            special_tokens = {"additional_special_tokens": ["[DIALECT]", "[INFORMAL]"]}
            model.add_special_tokens(special_tokens)
            
            model.tokenizer.add_special_tokens.assert_called_once_with(special_tokens)
            mock_resize.assert_called_once_with(30002)
    
    def test_resize_token_embeddings(self, mock_model_components):
        """Test resizing token embeddings."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        model.resize_token_embeddings(35000)
        
        model.bert.resize_token_embeddings.assert_called_once_with(35000)
    
    @patch('torch.save')
    @patch('pathlib.Path.mkdir')
    def test_save_model(self, mock_mkdir, mock_torch_save, mock_model_components):
        """Test model saving."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        save_path = Path("/test/path")
        model.save_model(save_path)
        
        # Check that directories were created
        mock_mkdir.assert_called()
        
        # Check that model components were saved
        model.bert.save_pretrained.assert_called_once()
        model.tokenizer.save_pretrained.assert_called_once()
        mock_torch_save.assert_called_once()
    
    @patch('torch.load')
    def test_load_model(self, mock_torch_load, mock_model_components):
        """Test model loading."""
        mock_config = {
            'model_name': 'test-model',
            'num_classes': 5,
            'max_length': 512,
            'dropout_rate': 0.1,
            'freeze_bert': False,
            'hidden_size': 768
        }
        mock_torch_load.return_value = mock_config
        
        load_path = Path("/test/path")
        model = UrduBertModel.load_model(load_path, device=torch.device('cpu'))
        
        assert model.model_name == 'test-model'
        assert model.num_classes == 5
        assert model.max_length == 512
    
    def test_get_attention_weights(self, mock_model_components):
        """Test getting attention weights."""
        model = UrduBertModel(device=torch.device('cpu'))
        
        # Mock tokenizer and model outputs
        mock_encoded = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]])
        }
        model.tokenizer.return_value = mock_encoded
        model.tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', 'test', 'token', '[SEP]']
        
        # Mock attention outputs
        mock_attention = torch.randn(1, 12, 5, 5)  # batch, heads, seq, seq
        mock_output = Mock()
        mock_output.attentions = [mock_attention]  # List of attention layers
        model.bert.return_value = mock_output
        
        attention_weights, tokens = model.get_attention_weights("test text")
        
        assert attention_weights.shape == (12, 5, 5)  # heads, seq, seq
        assert len(tokens) == 4
        assert tokens[0] == '[CLS]'