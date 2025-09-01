"""Tests for script normalization functionality."""

import pytest
from src.data.script_normalizer import ScriptNormalizer

class TestScriptNormalizer:
    """Test cases for ScriptNormalizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = ScriptNormalizer()
    
    def test_detect_script_type_urdu(self):
        """Test detection of Urdu script."""
        text = "یہ اردو متن ہے"
        script_type = self.normalizer.detect_script_type(text)
        assert script_type == 'urdu'
    
    def test_detect_script_type_roman(self):
        """Test detection of Roman script."""
        text = "yeh roman urdu hai"
        script_type = self.normalizer.detect_script_type(text)
        assert script_type == 'roman'
    
    def test_detect_script_type_mixed(self):
        """Test detection of mixed script."""
        text = "یہ mixed script ہے"
        script_type = self.normalizer.detect_script_type(text)
        assert script_type == 'mixed'
    
    def test_detect_script_type_empty(self):
        """Test detection with empty text."""
        script_type = self.normalizer.detect_script_type("")
        assert script_type == 'other'
    
    def test_is_roman_urdu_word_known_words(self):
        """Test Roman Urdu word detection for known words."""
        assert self.normalizer.is_roman_urdu_word("aur") is True
        assert self.normalizer.is_roman_urdu_word("hai") is True
        assert self.normalizer.is_roman_urdu_word("kya") is True
        assert self.normalizer.is_roman_urdu_word("main") is True
    
    def test_is_roman_urdu_word_patterns(self):
        """Test Roman Urdu word detection using patterns."""
        assert self.normalizer.is_roman_urdu_word("khaana") is True  # Contains 'kh'
        assert self.normalizer.is_roman_urdu_word("ghar") is True    # Contains 'gh'
        assert self.normalizer.is_roman_urdu_word("shaadi") is True  # Contains 'sh'
        assert self.normalizer.is_roman_urdu_word("chacha") is True  # Contains 'ch'
    
    def test_is_roman_urdu_word_english(self):
        """Test that English words are not detected as Roman Urdu."""
        assert self.normalizer.is_roman_urdu_word("hello") is False
        assert self.normalizer.is_roman_urdu_word("world") is False
        assert self.normalizer.is_roman_urdu_word("computer") is False
    
    def test_convert_word_roman_to_urdu_known_words(self):
        """Test conversion of known Roman Urdu words."""
        assert self.normalizer.convert_word_roman_to_urdu("aur") == "اور"
        assert self.normalizer.convert_word_roman_to_urdu("hai") == "ہے"
        assert self.normalizer.convert_word_roman_to_urdu("kya") == "کیا"
        assert self.normalizer.convert_word_roman_to_urdu("main") == "میں"
        assert self.normalizer.convert_word_roman_to_urdu("acha") == "اچھا"
    
    def test_convert_word_roman_to_urdu_patterns(self):
        """Test conversion using transliteration patterns."""
        # Test common patterns
        result = self.normalizer.convert_word_roman_to_urdu("khaana")
        assert "خ" in result  # Should contain kh -> خ
        
        result = self.normalizer.convert_word_roman_to_urdu("ghar")
        assert "غ" in result  # Should contain gh -> غ
        
        result = self.normalizer.convert_word_roman_to_urdu("shaadi")
        assert "ش" in result  # Should contain sh -> ش
    
    def test_convert_roman_to_urdu_sentence(self):
        """Test conversion of full Roman Urdu sentences."""
        text = "main acha hun aur tum kaise ho"
        converted = self.normalizer.convert_roman_to_urdu(text)
        
        # Should convert known words
        assert "میں" in converted  # main -> میں
        assert "اچھا" in converted  # acha -> اچھا
        assert "اور" in converted   # aur -> اور
        assert "تم" in converted    # tum -> تم
        assert "کیسے" in converted  # kaise -> کیسے
        assert "ہو" in converted    # ho -> ہو (if mapped)
    
    def test_convert_roman_to_urdu_with_punctuation(self):
        """Test conversion preserving punctuation."""
        text = "kya hal hai?"
        converted = self.normalizer.convert_roman_to_urdu(text)
        
        assert "کیا" in converted
        assert "ہے" in converted
        assert "?" in converted  # Punctuation should be preserved
    
    def test_normalize_mixed_script(self):
        """Test normalization of mixed script text."""
        text = "یہ ek mixed script hai"
        normalized = self.normalizer.normalize_mixed_script(text)
        
        # Urdu parts should remain unchanged
        assert "یہ" in normalized
        
        # Roman Urdu parts should be converted
        assert "ایک" in normalized  # ek -> ایک
        assert "ہے" in normalized   # hai -> ہے
    
    def test_normalize_script_pure_urdu(self):
        """Test normalization of pure Urdu text."""
        text = "یہ خالص اردو متن ہے"
        normalized = self.normalizer.normalize_script(text)
        
        # Should remain unchanged
        assert normalized == text
    
    def test_normalize_script_pure_roman(self):
        """Test normalization of pure Roman Urdu text."""
        text = "yeh roman urdu hai"
        normalized = self.normalizer.normalize_script(text)
        
        # Should be converted to Urdu
        assert "یہ" in normalized or "یے" in normalized  # yeh
        assert "ہے" in normalized  # hai
    
    def test_normalize_script_force_conversion(self):
        """Test forced conversion mode."""
        text = "hello world aur kya hal hai"
        normalized = self.normalizer.normalize_script(text, force_conversion=True)
        
        # Should attempt to convert everything
        assert "اور" in normalized  # aur -> اور
        assert "کیا" in normalized  # kya -> کیا
        assert "ہے" in normalized   # hai -> ہے
    
    def test_get_script_statistics(self):
        """Test script statistics generation."""
        text = "یہ mixed script hai with اردو"
        stats = self.normalizer.get_script_statistics(text)
        
        assert stats['script_type'] == 'mixed'
        assert stats['urdu_chars'] > 0
        assert stats['latin_chars'] > 0
        assert stats['total_chars'] > 0
        assert 0 < stats['urdu_ratio'] < 1
        assert 0 < stats['latin_ratio'] < 1
        assert stats['roman_urdu_words'] > 0
        assert stats['total_words'] > 0
    
    def test_get_script_statistics_empty(self):
        """Test script statistics for empty text."""
        stats = self.normalizer.get_script_statistics("")
        
        assert stats['script_type'] == 'other'
        assert stats['urdu_chars'] == 0
        assert stats['latin_chars'] == 0
        assert stats['total_chars'] == 0
        assert stats['urdu_ratio'] == 0.0
        assert stats['latin_ratio'] == 0.0
        assert stats['roman_urdu_words'] == 0
        assert stats['total_words'] == 0
    
    def test_case_insensitive_conversion(self):
        """Test that conversion works regardless of case."""
        assert self.normalizer.convert_word_roman_to_urdu("AUR") == "اور"
        assert self.normalizer.convert_word_roman_to_urdu("Hai") == "ہے"
        assert self.normalizer.convert_word_roman_to_urdu("KYA") == "کیا"
    
    def test_preserve_non_urdu_words(self):
        """Test that non-Urdu words are preserved."""
        text = "main computer use karta hun"
        converted = self.normalizer.convert_roman_to_urdu(text)
        
        # Urdu words should be converted
        assert "میں" in converted    # main
        assert "کرتا" in converted   # karta
        
        # English words should be preserved
        assert "computer" in converted
        assert "use" in converted
    
    def test_handle_none_and_empty_input(self):
        """Test handling of None and empty inputs."""
        assert self.normalizer.normalize_script(None) is None
        assert self.normalizer.normalize_script("") == ""
        assert self.normalizer.convert_roman_to_urdu(None) is None
        assert self.normalizer.convert_roman_to_urdu("") == ""