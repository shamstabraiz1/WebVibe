"""Tests for integrated text preprocessing pipeline."""

import pytest
from src.data.text_preprocessor import (
    TextPreprocessor, PreprocessingConfig, PreprocessingLevel,
    PreprocessingResult
)
from src.data.dialectal_normalizer import UrduDialect

class TestPreprocessingConfig:
    """Test cases for PreprocessingConfig."""
    
    def test_minimal_config(self):
        """Test minimal preprocessing configuration."""
        config = PreprocessingConfig.minimal()
        assert config.remove_urls is True
        assert config.normalize_script is False
        assert config.normalize_dialects is False
        assert config.enable_caching is False
    
    def test_standard_config(self):
        """Test standard preprocessing configuration."""
        config = PreprocessingConfig.standard()
        assert config.normalize_script is True
        assert config.normalize_dialects is True
        assert config.enable_caching is True
    
    def test_aggressive_config(self):
        """Test aggressive preprocessing configuration."""
        config = PreprocessingConfig.aggressive()
        assert config.remove_hashtags is True
        assert config.force_script_conversion is True
        assert config.normalize_dialects is True

class TestTextPreprocessor:
    """Test cases for TextPreprocessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        text = "یہ اچھا ہے! https://example.com @user"
        result = self.preprocessor.preprocess(text)
        
        assert isinstance(result, PreprocessingResult)
        assert result.original_text == text
        assert result.processed_text != text
        assert "https://example.com" not in result.processed_text
        assert "@user" not in result.processed_text
        assert "text_cleaning" in result.transformations_applied
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        result = self.preprocessor.preprocess("")
        
        assert result.processed_text == ""
        assert len(result.warnings) > 0
        assert "Empty or invalid input text" in result.warnings
    
    def test_none_text_handling(self):
        """Test handling of None text."""
        result = self.preprocessor.preprocess(None)
        
        assert result.processed_text == ""
        assert len(result.warnings) > 0
    
    def test_roman_urdu_processing(self):
        """Test processing of Roman Urdu text."""
        text = "yeh acha hai aur main khush hun"
        result = self.preprocessor.preprocess(text)
        
        assert result.original_script_type == "roman"
        assert "script_normalization" in result.transformations_applied
        # Should contain some Urdu script after conversion
        assert any(ord(char) >= 0x0600 for char in result.processed_text)
    
    def test_dialectal_processing(self):
        """Test processing of dialectal text."""
        text = "میں کیتا ہے اور ویلا آیا ہے"  # Punjabi-influenced
        result = self.preprocessor.preprocess(text)
        
        assert result.detected_dialect == UrduDialect.PUNJABI_INFLUENCED
        assert "dialectal_normalization" in result.transformations_applied
        assert "کیا" in result.processed_text  # کیتا -> کیا
        assert "والا" in result.processed_text  # ویلا -> والا
    
    def test_mixed_script_processing(self):
        """Test processing of mixed script text."""
        text = "یہ mixed script hai with اردو"
        result = self.preprocessor.preprocess(text)
        
        assert result.original_script_type == "mixed"
        assert "script_normalization" in result.transformations_applied
    
    def test_minimal_config_processing(self):
        """Test processing with minimal configuration."""
        config = PreprocessingConfig.minimal()
        preprocessor = TextPreprocessor(config)
        
        text = "yeh roman urdu hai with URLs https://example.com"
        result = preprocessor.preprocess(text)
        
        # Should only apply basic cleaning
        assert "text_cleaning" in result.transformations_applied
        assert "script_normalization" not in result.transformations_applied
        assert "dialectal_normalization" not in result.transformations_applied
    
    def test_aggressive_config_processing(self):
        """Test processing with aggressive configuration."""
        config = PreprocessingConfig.aggressive()
        preprocessor = TextPreprocessor(config)
        
        text = "yeh #test hai with @user and کیتا"
        result = preprocessor.preprocess(text)
        
        # Should apply all transformations
        assert "text_cleaning" in result.transformations_applied
        assert "script_normalization" in result.transformations_applied
        assert "dialectal_normalization" in result.transformations_applied
        assert "#test" not in result.processed_text  # Hashtags removed
    
    def test_batch_processing(self):
        """Test batch text processing."""
        texts = [
            "یہ اچھا ہے",
            "yeh roman hai",
            "میں کیتا ہے"
        ]
        results = self.preprocessor.preprocess_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, PreprocessingResult) for r in results)
        assert results[0].original_script_type == "urdu"
        assert results[1].original_script_type == "roman"
        assert results[2].detected_dialect == UrduDialect.PUNJABI_INFLUENCED
    
    def test_caching_functionality(self):
        """Test caching functionality."""
        config = PreprocessingConfig(enable_caching=True, cache_size=10)
        preprocessor = TextPreprocessor(config)
        
        text = "یہ ٹیسٹ ہے"
        
        # First call - cache miss
        result1 = preprocessor.preprocess(text)
        cache_stats = preprocessor.get_cache_stats()
        assert cache_stats['cache_misses'] == 1
        assert cache_stats['cache_hits'] == 0
        
        # Second call - cache hit
        result2 = preprocessor.preprocess(text)
        cache_stats = preprocessor.get_cache_stats()
        assert cache_stats['cache_hits'] == 1
        
        # Results should be identical
        assert result1.processed_text == result2.processed_text
    
    def test_cache_size_management(self):
        """Test cache size management."""
        config = PreprocessingConfig(enable_caching=True, cache_size=2)
        preprocessor = TextPreprocessor(config)
        
        # Add more items than cache size
        for i in range(5):
            preprocessor.preprocess(f"ٹیسٹ {i}")
        
        cache_stats = preprocessor.get_cache_stats()
        assert cache_stats['cache_size'] <= config.cache_size
    
    def test_cache_clearing(self):
        """Test cache clearing."""
        config = PreprocessingConfig(enable_caching=True)
        preprocessor = TextPreprocessor(config)
        
        preprocessor.preprocess("ٹیسٹ")
        assert preprocessor.get_cache_stats()['cache_size'] > 0
        
        preprocessor.clear_cache()
        cache_stats = preprocessor.get_cache_stats()
        assert cache_stats['cache_size'] == 0
        assert cache_stats['cache_hits'] == 0
        assert cache_stats['cache_misses'] == 0
    
    def test_config_update(self):
        """Test configuration update."""
        preprocessor = TextPreprocessor()
        
        # Update to minimal config
        new_config = PreprocessingConfig.minimal()
        preprocessor.update_config(new_config)
        
        assert preprocessor.config == new_config
        # Cache should be cleared after config update
        assert preprocessor.get_cache_stats()['cache_size'] == 0
    
    def test_text_analysis(self):
        """Test text analysis without preprocessing."""
        text = "یہ mixed script hai with کیتا"
        analysis = self.preprocessor.analyze_text(text)
        
        assert 'text_stats' in analysis
        assert 'script_type' in analysis
        assert 'script_stats' in analysis
        assert 'detected_dialect' in analysis
        assert 'dialectal_features' in analysis
        assert 'normalization_suggestions' in analysis
        
        assert analysis['script_type'] == 'mixed'
        assert analysis['detected_dialect'] == UrduDialect.PUNJABI_INFLUENCED
    
    def test_suggestions_generation(self):
        """Test generation of preprocessing suggestions."""
        text = "yeh roman hai with URLs https://example.com and @mentions"
        result = self.preprocessor.preprocess(text)
        
        assert len(result.suggestions) > 0
        # Should suggest script conversion and URL removal
        suggestion_text = ' '.join(result.suggestions)
        assert "Roman Urdu" in suggestion_text or "script" in suggestion_text
        assert "URL" in suggestion_text or "email" in suggestion_text
    
    def test_warnings_generation(self):
        """Test generation of preprocessing warnings."""
        # Test with text that becomes empty after processing
        text = "https://example.com @user #hashtag"
        config = PreprocessingConfig(
            remove_urls=True,
            remove_mentions=True,
            remove_hashtags=True,
            remove_special_chars=True
        )
        preprocessor = TextPreprocessor(config)
        result = preprocessor.preprocess(text)
        
        assert len(result.warnings) > 0
        warning_text = ' '.join(result.warnings)
        assert "empty" in warning_text.lower()
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        text = "یہ ٹیسٹ ہے with URLs https://example.com and @user"
        result = self.preprocessor.preprocess(text)
        
        assert 'total_chars' in result.original_stats
        assert 'total_words' in result.original_stats
        assert 'urls' in result.original_stats
        assert 'mentions' in result.original_stats
        
        assert 'total_chars' in result.processed_stats
        assert 'total_words' in result.processed_stats
        
        # Processed text should have fewer URLs and mentions
        assert result.processed_stats['urls'] <= result.original_stats['urls']
        assert result.processed_stats['mentions'] <= result.original_stats['mentions']
    
    def test_processing_time_tracking(self):
        """Test processing time tracking."""
        text = "یہ ٹیسٹ ہے"
        result = self.preprocessor.preprocess(text)
        
        assert result.processing_time > 0
        assert isinstance(result.processing_time, float)
    
    def test_transformation_tracking(self):
        """Test tracking of applied transformations."""
        text = "yeh roman hai aur کیتا ہے"
        result = self.preprocessor.preprocess(text)
        
        # Should track all applied transformations
        assert isinstance(result.transformations_applied, list)
        assert len(result.transformations_applied) > 0
        
        # Should include script and dialectal normalization
        assert "script_normalization" in result.transformations_applied
        assert "dialectal_normalization" in result.transformations_applied
    
    def test_preserve_urdu_content(self):
        """Test that Urdu content is preserved during preprocessing."""
        text = "یہ اردو متن ہے جو محفوظ رہنا چاہیے"
        result = self.preprocessor.preprocess(text)
        
        # Core Urdu words should be preserved
        assert "یہ" in result.processed_text
        assert "اردو" in result.processed_text
        assert "متن" in result.processed_text
        assert "ہے" in result.processed_text