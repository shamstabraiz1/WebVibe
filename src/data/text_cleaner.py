"""Text cleaning utilities for Urdu text preprocessing."""

import re
import unicodedata
from typing import List, Dict, Optional
from urllib.parse import urlparse

class UrduTextCleaner:
    """Text cleaner for Urdu text with support for social media content."""
    
    def __init__(self):
        """Initialize the text cleaner with patterns and mappings."""
        # URL patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.www_pattern = re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+')
        
        # Social media patterns
        self.mention_pattern = re.compile(r'@[a-zA-Z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[a-zA-Z0-9_\u0600-\u06FF]+')
        
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone number patterns (Pakistani formats)
        self.phone_pattern = re.compile(r'(\+92|0092|92)?[0-9]{10,11}')
        
        # Excessive punctuation and whitespace
        self.excessive_punct_pattern = re.compile(r'[۔!؟]{3,}')
        self.excessive_whitespace_pattern = re.compile(r'\s{2,}')
        
        # Repeated characters (more than 2 consecutive)
        self.repeated_char_pattern = re.compile(r'(.)\1{2,}')
        
        # Special characters to remove (keeping Urdu punctuation)
        self.special_chars_pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\s\w۔؍؎؏؞؟٪٫٬\.\,\!\?\:\;\'\"\-\(\)]')
        
        # Urdu digits to Arabic digits mapping
        self.urdu_to_arabic_digits = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
        }
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        text = self.url_pattern.sub(' ', text)
        text = self.www_pattern.sub(' ', text)
        return text
    
    def remove_mentions(self, text: str, replace_with: str = ' ') -> str:
        """Remove social media mentions."""
        return self.mention_pattern.sub(replace_with, text)
    
    def remove_hashtags(self, text: str, replace_with: str = ' ') -> str:
        """Remove hashtags (keeping the content after #)."""
        return self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses."""
        return self.email_pattern.sub(' ', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers."""
        return self.phone_pattern.sub(' ', text)
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize excessive punctuation."""
        # Replace excessive punctuation with single occurrence
        text = self.excessive_punct_pattern.sub(lambda m: m.group(0)[0], text)
        return text
    
    def normalize_repeated_characters(self, text: str) -> str:
        """Normalize repeated characters (keep max 2 repetitions)."""
        return self.repeated_char_pattern.sub(r'\1\1', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace multiple whitespace with single space
        text = self.excessive_whitespace_pattern.sub(' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters while preserving Urdu text and basic punctuation."""
        if keep_punctuation:
            return self.special_chars_pattern.sub(' ', text)
        else:
            # More aggressive cleaning - remove all non-Urdu characters except spaces
            pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\s]')
            return pattern.sub(' ', text)
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to NFC form (canonical decomposition followed by canonical composition)
        text = unicodedata.normalize('NFC', text)
        return text
    
    def convert_urdu_digits(self, text: str) -> str:
        """Convert Urdu digits to Arabic digits."""
        for urdu_digit, arabic_digit in self.urdu_to_arabic_digits.items():
            text = text.replace(urdu_digit, arabic_digit)
        return text
    
    def handle_encoding_issues(self, text: str) -> str:
        """Handle common encoding issues in Urdu text."""
        # Replace common encoding artifacts
        encoding_fixes = {
            'Ø§': 'ا',  # Alif
            'Ø¨': 'ب',  # Beh
            'Ù¾': 'پ',  # Peh
            'Øª': 'ت',  # Teh
            'Ù¹': 'ٹ',  # Tteh
        }
        
        for wrong, correct in encoding_fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def clean_text(
        self, 
        text: str,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,  # Keep hashtag content by default
        remove_emails: bool = True,
        remove_phone_numbers: bool = True,
        normalize_punctuation: bool = True,
        normalize_repeated_chars: bool = True,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = True,
        normalize_unicode: bool = True,
        convert_digits: bool = True,
        fix_encoding: bool = True,
        keep_punctuation: bool = True
    ) -> str:
        """
        Comprehensive text cleaning pipeline.
        
        Args:
            text: Input text to clean
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove hashtags (if False, keeps content after #)
            remove_emails: Remove email addresses
            remove_phone_numbers: Remove phone numbers
            normalize_punctuation: Normalize excessive punctuation
            normalize_repeated_chars: Normalize repeated characters
            normalize_whitespace: Normalize whitespace
            remove_special_chars: Remove special characters
            normalize_unicode: Normalize Unicode
            convert_digits: Convert Urdu digits to Arabic
            fix_encoding: Fix common encoding issues
            keep_punctuation: Keep basic punctuation when removing special chars
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Handle encoding issues first
        if fix_encoding:
            text = self.handle_encoding_issues(text)
        
        # Unicode normalization
        if normalize_unicode:
            text = self.normalize_unicode(text)
        
        # Remove unwanted elements
        if remove_urls:
            text = self.remove_urls(text)
        
        if remove_emails:
            text = self.remove_emails(text)
        
        if remove_phone_numbers:
            text = self.remove_phone_numbers(text)
        
        if remove_mentions:
            text = self.remove_mentions(text)
        
        if remove_hashtags:
            text = self.remove_hashtags(text)
        
        # Normalize text patterns
        if normalize_repeated_chars:
            text = self.normalize_repeated_characters(text)
        
        if normalize_punctuation:
            text = self.normalize_punctuation(text)
        
        if remove_special_chars:
            text = self.remove_special_characters(text, keep_punctuation)
        
        if convert_digits:
            text = self.convert_urdu_digits(text)
        
        # Final whitespace normalization
        if normalize_whitespace:
            text = self.normalize_whitespace(text)
        
        return text
    
    def get_text_stats(self, text: str) -> Dict[str, int]:
        """Get statistics about the text."""
        return {
            'total_chars': len(text),
            'total_words': len(text.split()),
            'urdu_chars': len(re.findall(r'[\u0600-\u06FF\u0750-\u077F]', text)),
            'urls': len(self.url_pattern.findall(text)),
            'mentions': len(self.mention_pattern.findall(text)),
            'hashtags': len(self.hashtag_pattern.findall(text)),
            'emails': len(self.email_pattern.findall(text)),
            'phone_numbers': len(self.phone_pattern.findall(text))
        }