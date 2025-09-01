"""Tests for text cleaner functionality."""

import pytest
from src.data.text_cleaner import UrduTextCleaner

class TestUrduTextCleaner:
    """Test cases for UrduTextCleaner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = UrduTextCleaner()
    
    def test_remove_urls(self):
        """Test URL removal."""
        text = "یہ ویب سائٹ https://example.com بہت اچھی ہے"
        cleaned = self.cleaner.remove_urls(text)
        assert "https://example.com" not in cleaned
        assert "یہ ویب سائٹ" in cleaned
        assert "بہت اچھی ہے" in cleaned
    
    def test_remove_www_urls(self):
        """Test www URL removal."""
        text = "دیکھیں www.example.com پر"
        cleaned = self.cleaner.remove_urls(text)
        assert "www.example.com" not in cleaned
        assert "دیکھیں" in cleaned
        assert "پر" in cleaned
    
    def test_remove_mentions(self):
        """Test mention removal."""
        text = "سلام @user123 کیا حال ہے؟"
        cleaned = self.cleaner.remove_mentions(text)
        assert "@user123" not in cleaned
        assert "سلام" in cleaned
        assert "کیا حال ہے؟" in cleaned
    
    def test_remove_hashtags_keep_content(self):
        """Test hashtag removal while keeping content."""
        text = "آج #پاکستان میں بارش ہے"
        cleaned = self.cleaner.remove_hashtags(text)
        assert "#پاکستان" not in cleaned
        assert "پاکستان" in cleaned
        assert "آج" in cleaned
        assert "میں بارش ہے" in cleaned
    
    def test_remove_emails(self):
        """Test email removal."""
        text = "رابطہ کریں test@example.com پر"
        cleaned = self.cleaner.remove_emails(text)
        assert "test@example.com" not in cleaned
        assert "رابطہ کریں" in cleaned
        assert "پر" in cleaned
    
    def test_remove_phone_numbers(self):
        """Test phone number removal."""
        text = "کال کریں 03001234567 پر"
        cleaned = self.cleaner.remove_phone_numbers(text)
        assert "03001234567" not in cleaned
        assert "کال کریں" in cleaned
        assert "پر" in cleaned
    
    def test_normalize_punctuation(self):
        """Test punctuation normalization."""
        text = "واہ!!!!! کیا بات ہے۔۔۔۔"
        cleaned = self.cleaner.normalize_punctuation(text)
        assert "!!!!!" not in cleaned
        assert "۔۔۔۔" not in cleaned
        assert "!" in cleaned
        assert "۔" in cleaned
    
    def test_normalize_repeated_characters(self):
        """Test repeated character normalization."""
        text = "واااااہ کیااااا بات ہے"
        cleaned = self.cleaner.normalize_repeated_characters(text)
        assert "واااااہ" not in cleaned
        assert "کیااااا" not in cleaned
        assert "واہ" in cleaned or "وااہ" in cleaned  # Max 2 repetitions
        assert "کیا" in cleaned or "کیاا" in cleaned
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "  یہ    بہت   اچھا     ہے  "
        cleaned = self.cleaner.normalize_whitespace(text)
        assert cleaned == "یہ بہت اچھا ہے"
        assert "    " not in cleaned
    
    def test_convert_urdu_digits(self):
        """Test Urdu to Arabic digit conversion."""
        text = "سال ۲۰۲۳ میں ۱۰ کتابیں پڑھیں"
        cleaned = self.cleaner.convert_urdu_digits(text)
        assert "۲۰۲۳" not in cleaned
        assert "۱۰" not in cleaned
        assert "2023" in cleaned
        assert "10" in cleaned
    
    def test_handle_encoding_issues(self):
        """Test encoding issue handling."""
        text = "Ø§Ø³ Ù¾Ø± Ø¨Ø§Øª"  # Corrupted Urdu text
        cleaned = self.cleaner.handle_encoding_issues(text)
        # Should fix some common encoding issues
        assert cleaned != text
    
    def test_comprehensive_cleaning(self):
        """Test comprehensive text cleaning."""
        text = """
        سلام @user123 ! یہ ویب سائٹ https://example.com پر دیکھیں۔۔۔
        رابطہ: test@example.com یا 03001234567
        #پاکستان میں آج بہت گرمی ہے!!!!
        سال ۲۰۲۳ میں واااہ کیا موسم ہے    
        """
        
        cleaned = self.cleaner.clean_text(text)
        
        # Should remove unwanted elements
        assert "@user123" not in cleaned
        assert "https://example.com" not in cleaned
        assert "test@example.com" not in cleaned
        assert "03001234567" not in cleaned
        
        # Should normalize patterns
        assert "!!!!" not in cleaned
        assert "۔۔۔" not in cleaned
        assert "واااہ" not in cleaned
        assert "    " not in cleaned
        
        # Should keep Urdu content
        assert "سلام" in cleaned
        assert "پاکستان" in cleaned
        assert "گرمی" in cleaned
        
        # Should convert digits
        assert "2023" in cleaned
    
    def test_empty_text_handling(self):
        """Test handling of empty or None text."""
        assert self.cleaner.clean_text("") == ""
        assert self.cleaner.clean_text(None) == ""
        assert self.cleaner.clean_text("   ") == ""
    
    def test_text_stats(self):
        """Test text statistics generation."""
        text = "سلام @user https://example.com #test test@email.com 03001234567"
        stats = self.cleaner.get_text_stats(text)
        
        assert stats['urls'] == 1
        assert stats['mentions'] == 1
        assert stats['hashtags'] == 1
        assert stats['emails'] == 1
        assert stats['phone_numbers'] == 1
        assert stats['total_words'] > 0
        assert stats['urdu_chars'] > 0
    
    def test_selective_cleaning(self):
        """Test selective cleaning options."""
        text = "سلام @user! یہ #test ہے https://example.com"
        
        # Only remove URLs
        cleaned = self.cleaner.clean_text(
            text,
            remove_urls=True,
            remove_mentions=False,
            remove_hashtags=False,
            remove_special_chars=False
        )
        assert "https://example.com" not in cleaned
        assert "@user" in cleaned
        assert "#test" in cleaned
        
        # Only remove mentions
        cleaned = self.cleaner.clean_text(
            text,
            remove_urls=False,
            remove_mentions=True,
            remove_hashtags=False,
            remove_special_chars=False
        )
        assert "https://example.com" in cleaned
        assert "@user" not in cleaned
        assert "#test" in cleaned
    
    def test_keep_punctuation_option(self):
        """Test keep punctuation option."""
        text = "یہ اچھا ہے! کیا بات ہے؟"
        
        # Keep punctuation
        cleaned = self.cleaner.clean_text(text, keep_punctuation=True)
        assert "!" in cleaned
        assert "؟" in cleaned
        
        # Remove punctuation
        cleaned = self.cleaner.clean_text(text, keep_punctuation=False)
        # Basic punctuation might still be there, but special chars removed
        assert "یہ اچھا ہے" in cleaned