"""Tests for dialectal normalization functionality."""

import pytest
from src.data.dialectal_normalizer import DialectalNormalizer, UrduDialect

class TestDialectalNormalizer:
    """Test cases for DialectalNormalizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = DialectalNormalizer()
    
    def test_detect_dialect_standard(self):
        """Test detection of standard Urdu."""
        text = "یہ معیاری اردو متن ہے"
        dialect = self.normalizer.detect_dialect(text)
        assert dialect == UrduDialect.STANDARD
    
    def test_detect_dialect_punjabi(self):
        """Test detection of Punjabi-influenced Urdu."""
        text = "میں کیتا ہے اور ویلا آیا ہے"
        dialect = self.normalizer.detect_dialect(text)
        assert dialect == UrduDialect.PUNJABI_INFLUENCED
    
    def test_detect_dialect_sindhi(self):
        """Test detection of Sindhi-influenced Urdu."""
        text = "هن ڪار ڪري رهيو آهي"
        dialect = self.normalizer.detect_dialect(text)
        assert dialect == UrduDialect.SINDHI_INFLUENCED
    
    def test_detect_dialect_pashto(self):
        """Test detection of Pashto-influenced Urdu."""
        text = "زه کور ته ځم"
        dialect = self.normalizer.detect_dialect(text)
        assert dialect == UrduDialect.PASHTO_INFLUENCED
    
    def test_detect_dialect_dakhini(self):
        """Test detection of Dakhini Urdu."""
        text = "کیکر حال ہے اور کیدر جا رہے ہو"
        dialect = self.normalizer.detect_dialect(text)
        assert dialect == UrduDialect.DAKHINI
    
    def test_detect_dialect_empty(self):
        """Test detection with empty text."""
        dialect = self.normalizer.detect_dialect("")
        assert dialect == UrduDialect.UNKNOWN
    
    def test_normalize_punjabi_variations(self):
        """Test normalization of Punjabi variations."""
        text = "میں کیتا ہے اور ویلا آیا ہے"
        normalized = self.normalizer.normalize_specific_dialect(text, UrduDialect.PUNJABI_INFLUENCED)
        
        assert "کیا" in normalized  # کیتا -> کیا
        assert "والا" in normalized  # ویلا -> والا
        assert "کیتا" not in normalized
        assert "ویلا" not in normalized
    
    def test_normalize_sindhi_variations(self):
        """Test normalization of Sindhi variations."""
        text = "هن ڪار ڪري رهيو آهي"
        normalized = self.normalizer.normalize_specific_dialect(text, UrduDialect.SINDHI_INFLUENCED)
        
        # Should normalize Sindhi characters
        assert "ک" in normalized  # ڪ -> ک
        assert "ڪ" not in normalized
    
    def test_normalize_pashto_variations(self):
        """Test normalization of Pashto variations."""
        text = "زه کور ته ځم"
        normalized = self.normalizer.normalize_specific_dialect(text, UrduDialect.PASHTO_INFLUENCED)
        
        assert "میں" in normalized  # زه -> میں
        assert "گھر" in normalized  # کور -> گھر
        assert "تو" in normalized   # ته -> تو
        assert "ز" in normalized    # ځ -> ز
    
    def test_normalize_dakhini_variations(self):
        """Test normalization of Dakhini variations."""
        text = "کیکر حال ہے اور کیدر جا رہے ہو"
        normalized = self.normalizer.normalize_specific_dialect(text, UrduDialect.DAKHINI)
        
        assert "کیسے" in normalized  # کیکر -> کیسے
        assert "کہاں" in normalized  # کیدر -> کہاں
        assert "کیکر" not in normalized
        assert "کیدر" not in normalized
    
    def test_normalize_informal_variations(self):
        """Test normalization of informal/slang variations."""
        text = "کیہ حال ہ یار"
        normalized = self.normalizer.normalize_dialect(text)
        
        assert "کیا" in normalized   # کیہ -> کیا
        assert "ہے" in normalized    # ہ -> ہے
        assert "دوست" in normalized  # یار -> دوست
    
    def test_normalize_standard_text_unchanged(self):
        """Test that standard text remains unchanged."""
        text = "یہ معیاری اردو متن ہے"
        normalized = self.normalizer.normalize_dialect(text)
        
        # Should remain the same
        assert normalized == text
    
    def test_get_dialectal_features(self):
        """Test extraction of dialectal features."""
        text = "میں کیتا ہے اور ویلا آیا ہے"
        features = self.normalizer.get_dialectal_features(text)
        
        assert len(features[UrduDialect.PUNJABI_INFLUENCED.value]) > 0
        assert "کیتا" in features[UrduDialect.PUNJABI_INFLUENCED.value]
        assert "ویلا" in features[UrduDialect.PUNJABI_INFLUENCED.value]
    
    def test_get_dialectal_features_empty(self):
        """Test dialectal features extraction with empty text."""
        features = self.normalizer.get_dialectal_features("")
        
        for dialect_features in features.values():
            assert len(dialect_features) == 0
    
    def test_get_normalization_suggestions(self):
        """Test normalization suggestions."""
        text = "کیہ حال ہ یار"
        suggestions = self.normalizer.get_normalization_suggestions(text)
        
        assert "کیہ" in suggestions
        assert suggestions["کیہ"] == "کیا"
        assert "ہ" in suggestions
        assert suggestions["ہ"] == "ہے"
        assert "یار" in suggestions
        assert suggestions["یار"] == "دوست"
    
    def test_get_normalization_suggestions_empty(self):
        """Test normalization suggestions with empty text."""
        suggestions = self.normalizer.get_normalization_suggestions("")
        assert len(suggestions) == 0
    
    def test_standardize_text_with_detection(self):
        """Test text standardization with automatic dialect detection."""
        text = "میں کیتا ہے اور ویلا آیا ہے"
        standardized = self.normalizer.standardize_text(text)
        
        assert "کیا" in standardized
        assert "والا" in standardized
        assert "کیتا" not in standardized
        assert "ویلا" not in standardized
    
    def test_standardize_text_with_provided_dialect(self):
        """Test text standardization with provided dialect."""
        text = "کیکر حال ہے"
        standardized = self.normalizer.standardize_text(text, UrduDialect.DAKHINI)
        
        assert "کیسے" in standardized
        assert "کیکر" not in standardized
    
    def test_standardize_text_empty(self):
        """Test standardization with empty text."""
        standardized = self.normalizer.standardize_text("")
        assert standardized == ""
    
    def test_word_boundary_matching(self):
        """Test that normalization uses word boundaries correctly."""
        # Test that partial matches are not replaced
        text = "کیتاب میں کیتا ہے"  # کیتاب should not be affected
        normalized = self.normalizer.normalize_specific_dialect(text, UrduDialect.PUNJABI_INFLUENCED)
        
        assert "کیتاب" in normalized  # Should remain unchanged
        assert "کیا" in normalized    # کیتا should be normalized
    
    def test_multiple_dialect_features(self):
        """Test text with features from multiple dialects."""
        text = "میں کیتا ہے اور کیکر حال ہے"  # Punjabi + Dakhini
        
        # Should detect the dialect with more features
        dialect = self.normalizer.detect_dialect(text)
        assert dialect in [UrduDialect.PUNJABI_INFLUENCED, UrduDialect.DAKHINI]
        
        # Standardization should handle both
        standardized = self.normalizer.standardize_text(text)
        assert "کیا" in standardized
        assert "کیسے" in standardized
    
    def test_case_sensitivity(self):
        """Test that normalization handles case properly."""
        # Most Urdu text doesn't have case variations, but test with available examples
        text = "یار کیا حال ہے"
        normalized = self.normalizer.normalize_dialect(text)
        
        assert "دوست" in normalized  # یار -> دوست
        assert "کیا" in normalized    # Should remain کیا
    
    def test_preserve_context(self):
        """Test that normalization preserves context and meaning."""
        text = "اوہ ویلا بہت اچھا ہے"
        normalized = self.normalizer.normalize_specific_dialect(text, UrduDialect.PUNJABI_INFLUENCED)
        
        assert "وہ" in normalized    # اوہ -> وہ
        assert "والا" in normalized  # ویلا -> والا
        assert "بہت اچھا ہے" in normalized  # Context preserved