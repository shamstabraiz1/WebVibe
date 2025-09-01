"""Script normalization utilities for Urdu text, handling Roman Urdu conversion."""

import re
from typing import Dict, List, Tuple, Optional

class ScriptNormalizer:
    """Normalizer for converting Roman Urdu to Urdu script and handling mixed scripts."""
    
    def __init__(self):
        """Initialize the script normalizer with mapping dictionaries."""
        
        # Roman to Urdu character mappings
        self.roman_to_urdu_chars = {
            # Basic consonants
            'b': 'ب', 'p': 'پ', 't': 'ت', 'T': 'ٹ',
            'j': 'ج', 'ch': 'چ', 'H': 'ح', 'kh': 'خ',
            'd': 'د', 'D': 'ڈ', 'z': 'ذ', 'r': 'ر',
            'R': 'ڑ', 's': 'س', 'sh': 'ش', 'S': 'ص',
            'Z': 'ض', 'x': 'ط', 'X': 'ظ', 'A': 'ع',
            'gh': 'غ', 'f': 'ف', 'q': 'ق', 'k': 'ک',
            'g': 'گ', 'l': 'ل', 'm': 'م', 'n': 'ن',
            'w': 'و', 'h': 'ہ', 'y': 'ی',
            
            # Vowels
            'a': 'ا', 'i': 'ی', 'u': 'و', 'e': 'ے',
            'o': 'و',
            
            # Special combinations
            'aa': 'آ', 'ai': 'ائ', 'au': 'او',
        }
        
        # Common Roman Urdu words to Urdu mappings
        self.roman_to_urdu_words = {
            # Common words
            'aur': 'اور', 'hai': 'ہے', 'hain': 'ہیں', 'ka': 'کا',
            'ki': 'کی', 'ke': 'کے', 'ko': 'کو', 'se': 'سے',
            'me': 'میں', 'mein': 'میں', 'par': 'پر', 'pe': 'پے',
            'is': 'اس', 'us': 'اس', 'ye': 'یہ', 'yeh': 'یہ',
            'wo': 'وہ', 'woh': 'وہ', 'kya': 'کیا', 'kyun': 'کیوں',
            'kaise': 'کیسے', 'kahan': 'کہاں', 'kab': 'کب',
            'kaun': 'کون', 'kitna': 'کتنا', 'kitni': 'کتنی',
            
            # Pronouns
            'main': 'میں', 'tu': 'تو', 'tum': 'تم', 'aap': 'آپ',
            'hum': 'ہم', 'tumhara': 'تمہارا', 'tumhari': 'تمہاری',
            'hamara': 'ہمارا', 'hamari': 'ہماری', 'uska': 'اسکا',
            'uski': 'اسکی', 'unka': 'انکا', 'unki': 'انکی',
            
            # Common verbs
            'karna': 'کرنا', 'karta': 'کرتا', 'karti': 'کرتی',
            'karte': 'کرتے', 'kiya': 'کیا', 'kiye': 'کیے',
            'jana': 'جانا', 'jata': 'جاتا', 'jati': 'جاتی',
            'jate': 'جاتے', 'gaya': 'گیا', 'gayi': 'گئی',
            'gaye': 'گئے', 'aana': 'آنا', 'ata': 'آتا',
            'ati': 'آتی', 'ate': 'آتے', 'aya': 'آیا',
            'ayi': 'آئی', 'aye': 'آئے', 'hona': 'ہونا',
            'hua': 'ہوا', 'hui': 'ہوئی', 'hue': 'ہوئے',
            'dekha': 'دیکھا', 'dekhi': 'دیکھی', 'dekhe': 'دیکھے',
            'suna': 'سنا', 'suni': 'سنی', 'sune': 'سنے',
            'kaha': 'کہا', 'kahi': 'کہی', 'kahe': 'کہے',
            
            # Common adjectives
            'acha': 'اچھا', 'achi': 'اچھی', 'ache': 'اچھے',
            'bura': 'برا', 'buri': 'بری', 'bure': 'برے',
            'bara': 'بڑا', 'bari': 'بڑی', 'bare': 'بڑے',
            'chota': 'چھوٹا', 'choti': 'چھوٹی', 'chote': 'چھوٹے',
            'naya': 'نیا', 'nayi': 'نئی', 'naye': 'نئے',
            'purana': 'پرانا', 'purani': 'پرانی', 'purane': 'پرانے',
            
            # Numbers
            'ek': 'ایک', 'do': 'دو', 'teen': 'تین', 'char': 'چار',
            'panch': 'پانچ', 'che': 'چھ', 'saat': 'سات', 'aath': 'آٹھ',
            'nau': 'نو', 'das': 'دس', 'gyarah': 'گیارہ', 'barah': 'بارہ',
            
            # Time and days
            'din': 'دن', 'raat': 'رات', 'subah': 'صبح', 'sham': 'شام',
            'peer': 'پیر', 'mangal': 'منگل', 'budh': 'بدھ', 'jumerat': 'جمعرات',
            'jumma': 'جمعہ', 'hafta': 'ہفتہ', 'itwaar': 'اتوار',
            
            # Common expressions
            'salam': 'سلام', 'namaste': 'نمسکار', 'alvida': 'الوداع',
            'shukriya': 'شکریہ', 'maaf': 'معاف', 'sorry': 'معذرت',
            'please': 'برائے کرم', 'welcome': 'خوش آمدید',
        }
        
        # Patterns for detecting Roman Urdu
        self.roman_urdu_patterns = [
            r'\b[a-zA-Z]+\b',  # Basic Latin words
            r'\b[a-zA-Z]+[0-9]*\b',  # Words with numbers
        ]
        
        # Patterns for detecting mixed script
        self.urdu_char_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')
        self.latin_char_pattern = re.compile(r'[a-zA-Z]')
        
        # Common transliteration patterns
        self.transliteration_patterns = [
            (r'kh', 'خ'), (r'gh', 'غ'), (r'sh', 'ش'), (r'ch', 'چ'),
            (r'th', 'تھ'), (r'dh', 'دھ'), (r'ph', 'پھ'), (r'bh', 'بھ'),
            (r'aa', 'آ'), (r'ee', 'ی'), (r'oo', 'و'), (r'ai', 'ائ'),
            (r'au', 'او'), (r'ng', 'نگ'), (r'nk', 'نک'),
        ]
    
    def detect_script_type(self, text: str) -> str:
        """
        Detect the script type of the text.
        
        Returns:
            'urdu': Primarily Urdu script
            'roman': Primarily Roman/Latin script
            'mixed': Mixed scripts
            'other': Other scripts or unclear
        """
        if not text or not text.strip():
            return 'other'
        
        urdu_chars = len(self.urdu_char_pattern.findall(text))
        latin_chars = len(self.latin_char_pattern.findall(text))
        total_chars = urdu_chars + latin_chars
        
        if total_chars == 0:
            return 'other'
        
        urdu_ratio = urdu_chars / total_chars
        latin_ratio = latin_chars / total_chars
        
        if urdu_ratio > 0.8:
            return 'urdu'
        elif latin_ratio > 0.8:
            return 'roman'
        elif urdu_ratio > 0.2 and latin_ratio > 0.2:
            return 'mixed'
        else:
            return 'other'
    
    def is_roman_urdu_word(self, word: str) -> bool:
        """Check if a word is likely Roman Urdu."""
        word_lower = word.lower()
        
        # Check if it's in our known Roman Urdu words
        if word_lower in self.roman_to_urdu_words:
            return True
        
        # Check if it matches Roman Urdu patterns
        if re.match(r'^[a-zA-Z]+$', word):
            # Additional heuristics for Roman Urdu detection
            # Check for common Urdu phonetic patterns
            urdu_patterns = ['kh', 'gh', 'sh', 'ch', 'aa', 'ee', 'oo']
            for pattern in urdu_patterns:
                if pattern in word_lower:
                    return True
        
        return False
    
    def convert_word_roman_to_urdu(self, word: str) -> str:
        """Convert a single Roman Urdu word to Urdu script."""
        word_lower = word.lower()
        
        # Direct word mapping
        if word_lower in self.roman_to_urdu_words:
            return self.roman_to_urdu_words[word_lower]
        
        # Character-by-character conversion for unknown words
        result = word_lower
        
        # Apply transliteration patterns (longer patterns first)
        for pattern, replacement in sorted(self.transliteration_patterns, 
                                         key=lambda x: len(x[0]), reverse=True):
            result = re.sub(pattern, replacement, result)
        
        # Apply single character mappings
        for roman_char, urdu_char in self.roman_to_urdu_chars.items():
            if len(roman_char) == 1:  # Only single characters
                result = result.replace(roman_char, urdu_char)
        
        return result
    
    def convert_roman_to_urdu(self, text: str) -> str:
        """Convert Roman Urdu text to Urdu script."""
        if not text:
            return text
        
        words = text.split()
        converted_words = []
        
        for word in words:
            # Remove punctuation for processing
            clean_word = re.sub(r'[^\w]', '', word)
            punctuation = re.sub(r'[\w]', '', word)
            
            if self.is_roman_urdu_word(clean_word):
                converted_word = self.convert_word_roman_to_urdu(clean_word)
                converted_words.append(converted_word + punctuation)
            else:
                converted_words.append(word)
        
        return ' '.join(converted_words)
    
    def normalize_mixed_script(self, text: str) -> str:
        """Normalize text with mixed scripts."""
        script_type = self.detect_script_type(text)
        
        if script_type == 'roman':
            return self.convert_roman_to_urdu(text)
        elif script_type == 'mixed':
            # For mixed script, convert only Roman Urdu parts
            words = text.split()
            normalized_words = []
            
            for word in words:
                # Check if word contains only Latin characters
                if re.match(r'^[a-zA-Z\W]*$', word):
                    clean_word = re.sub(r'[^\w]', '', word)
                    if self.is_roman_urdu_word(clean_word):
                        punctuation = re.sub(r'[\w]', '', word)
                        converted = self.convert_word_roman_to_urdu(clean_word)
                        normalized_words.append(converted + punctuation)
                    else:
                        normalized_words.append(word)
                else:
                    normalized_words.append(word)
            
            return ' '.join(normalized_words)
        
        return text
    
    def normalize_script(self, text: str, force_conversion: bool = False) -> str:
        """
        Main script normalization function.
        
        Args:
            text: Input text to normalize
            force_conversion: If True, attempts conversion even for uncertain cases
            
        Returns:
            Normalized text with consistent script
        """
        if not text or not isinstance(text, str):
            return text
        
        if force_conversion:
            return self.convert_roman_to_urdu(text)
        else:
            return self.normalize_mixed_script(text)
    
    def get_script_statistics(self, text: str) -> Dict[str, any]:
        """Get statistics about script usage in text."""
        if not text:
            return {
                'script_type': 'other',
                'urdu_chars': 0,
                'latin_chars': 0,
                'total_chars': 0,
                'urdu_ratio': 0.0,
                'latin_ratio': 0.0,
                'roman_urdu_words': 0,
                'total_words': 0
            }
        
        urdu_chars = len(self.urdu_char_pattern.findall(text))
        latin_chars = len(self.latin_char_pattern.findall(text))
        total_chars = urdu_chars + latin_chars
        
        words = text.split()
        roman_urdu_words = sum(1 for word in words if self.is_roman_urdu_word(word))
        
        return {
            'script_type': self.detect_script_type(text),
            'urdu_chars': urdu_chars,
            'latin_chars': latin_chars,
            'total_chars': total_chars,
            'urdu_ratio': urdu_chars / total_chars if total_chars > 0 else 0.0,
            'latin_ratio': latin_chars / total_chars if total_chars > 0 else 0.0,
            'roman_urdu_words': roman_urdu_words,
            'total_words': len(words)
        }