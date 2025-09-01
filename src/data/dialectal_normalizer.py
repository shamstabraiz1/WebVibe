"""Dialectal normalization utilities for handling Urdu dialect variations."""

import re
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum

class UrduDialect(Enum):
    """Urdu dialect types."""
    STANDARD = "standard"
    PUNJABI_INFLUENCED = "punjabi_influenced"
    SINDHI_INFLUENCED = "sindhi_influenced"
    PASHTO_INFLUENCED = "pashto_influenced"
    BALOCHI_INFLUENCED = "balochi_influenced"
    DAKHINI = "dakhini"
    UNKNOWN = "unknown"

class DialectalNormalizer:
    """Normalizer for handling dialectal variations in Urdu text."""
    
    def __init__(self):
        """Initialize the dialectal normalizer with mapping dictionaries."""
        
        # Punjabi-influenced Urdu normalizations
        self.punjabi_normalizations = {
            # Phonetic variations
            'ویلا': 'والا',      # wala variations
            'ویلے': 'والے',
            'ویلی': 'والی',
            'کیتا': 'کیا',       # kita -> kya
            'کیتی': 'کی',        # kiti -> ki
            'کیتے': 'کے',        # kite -> ke
            'ہوئیا': 'ہوا',      # hoyia -> hua
            'گئیا': 'گیا',       # gayia -> gaya
            'آئیا': 'آیا',       # ayia -> aya
            'لئی': 'لیے',        # layi -> liye
            'نئیں': 'نہیں',      # nayi -> nahi
            'کوئی': 'کوئی',       # koyi (already standard)
            'ہوئے': 'ہوئے',      # hoye (already standard)
            
            # Vocabulary variations
            'پانی': 'پانی',       # pani (standard)
            'جل': 'پانی',         # jal -> pani
            'گھر': 'گھر',         # ghar (standard)
            'کوٹھی': 'گھر',       # kothi -> ghar
            'کھانا': 'کھانا',     # khana (standard)
            'روٹی': 'روٹی',       # roti (standard)
            'پھلکا': 'روٹی',      # phulka -> roti
            
            # Pronouns and particles
            'میں': 'میں',         # main (standard)
            'میں نوں': 'مجھے',    # main nu -> mujhe
            'تیں': 'تم',          # tain -> tum
            'اوہ': 'وہ',          # oh -> woh
            'ایہ': 'یہ',          # eh -> yeh
            'کی': 'کیا',          # ki -> kya (in questions)
        }
        
        # Sindhi-influenced Urdu normalizations
        self.sindhi_normalizations = {
            # Phonetic variations
            'ڪ': 'ک',            # Sindhi kaf -> Urdu kaf
            'ڳ': 'گ',            # Sindhi gaf -> Urdu gaf
            'ڄ': 'ج',            # Sindhi jeem -> Urdu jeem
            'ڇ': 'چ',            # Sindhi che -> Urdu che
            'ڏ': 'د',            # Sindhi dal -> Urdu dal
            'ڙ': 'ڑ',            # Sindhi rre (similar to Urdu)
            'ڻ': 'ن',            # Sindhi noon -> Urdu noon
            
            # Vocabulary variations
            'پاڻي': 'پانی',       # pani variations
            'گھر': 'گھر',         # ghar (standard)
            'ڪار': 'کام',         # kar -> kam
            'هن': 'اس',          # hun -> us
            'هو': 'وہ',          # ho -> woh
            'اهو': 'یہ',         # aho -> yeh
            
            # Common expressions
            'ڪيئن': 'کیسے',      # kiyan -> kaise
            'ڪٿي': 'کہاں',       # kithi -> kahan
            'ڪڏهن': 'کب',        # kadhan -> kab
        }
        
        # Pashto-influenced Urdu normalizations
        self.pashto_normalizations = {
            # Phonetic variations
            'ښ': 'ش',            # Pashto xa -> Urdu sheen
            'ځ': 'ز',            # Pashto ze -> Urdu zal
            'څ': 'ص',            # Pashto tse -> Urdu swad
            'ډ': 'ڈ',            # Pashto ddal -> Urdu ddal
            'ړ': 'ڑ',            # Pashto rre -> Urdu rre
            'ږ': 'ژ',            # Pashto zhe -> Urdu zhe
            'ګ': 'گ',            # Pashto gaf -> Urdu gaf
            'ڼ': 'ن',            # Pashto noon -> Urdu noon
            
            # Vocabulary variations
            'اوبه': 'پانی',       # obah -> pani
            'کور': 'گھر',        # kor -> ghar
            'ډوډۍ': 'روٹی',      # dodai -> roti
            'زه': 'میں',         # za -> main
            'ته': 'تو',          # ta -> tu
            'هغه': 'وہ',         # hagha -> woh
            'دا': 'یہ',          # da -> yeh
            
            # Common expressions
            'څنګه': 'کیسے',      # tsanga -> kaise
            'چېرته': 'کہاں',     # cherta -> kahan
            'کله': 'کب',         # kala -> kab
        }
        
        # Balochi-influenced Urdu normalizations
        self.balochi_normalizations = {
            # Phonetic variations (limited, as Balochi uses different script)
            'آپ': 'آپ',          # aap (standard)
            'شما': 'آپ',         # shuma -> aap
            'من': 'میں',         # man -> main
            'تو': 'تو',          # to (standard)
            'او': 'وہ',          # o -> woh
            'ای': 'یہ',          # e -> yeh
            
            # Common words
            'آب': 'پانی',        # aab -> pani
            'خانه': 'گھر',       # khana -> ghar
            'نان': 'روٹی',       # nan -> roti
        }
        
        # Dakhini (Deccan) Urdu normalizations
        self.dakhini_normalizations = {
            # Characteristic Dakhini features
            'کون': 'کون',        # kaun (standard)
            'کونسا': 'کونسا',    # kaunsa (standard)
            'کیکر': 'کیسے',      # kaikar -> kaise
            'کیکو': 'کیوں',      # kaiko -> kyun
            'کیدر': 'کہاں',      # kaidar -> kahan
            'کیدو': 'کب',        # kaido -> kab
            
            # Vocabulary variations
            'پانی': 'پانی',       # pani (standard)
            'نیر': 'پانی',        # neer -> pani
            'گھر': 'گھر',         # ghar (standard)
            'مکان': 'گھر',        # makan -> ghar
            'کھانا': 'کھانا',     # khana (standard)
            'آہار': 'کھانا',      # ahar -> khana
            
            # Pronouns
            'میں': 'میں',         # main (standard)
            'ہم': 'ہم',          # hum (standard)
            'تم': 'تم',          # tum (standard)
            'آپ': 'آپ',          # aap (standard)
        }
        
        # Informal/slang normalizations (common across dialects)
        self.informal_normalizations = {
            # Shortened forms
            'کیا': 'کیا',         # kya (standard)
            'کیہ': 'کیا',         # kyah -> kya
            'کیے': 'کیا',         # kiye -> kya (in some contexts)
            'ہے': 'ہے',          # hai (standard)
            'ہ': 'ہے',           # h -> hai
            'نہیں': 'نہیں',       # nahi (standard)
            'نہ': 'نہیں',         # na -> nahi
            'نئیں': 'نہیں',       # nayi -> nahi
            
            # Common abbreviations
            'اور': 'اور',         # aur (standard)
            'ر': 'اور',           # r -> aur
            'کے': 'کے',          # ke (standard)
            'ک': 'کے',           # k -> ke
            'میں': 'میں',         # mein (standard)
            'م': 'میں',          # m -> mein
            
            # Internet slang
            'یار': 'دوست',       # yaar -> dost
            'بھائی': 'بھائی',     # bhai (standard)
            'بھیا': 'بھائی',      # bhiya -> bhai
            'آپا': 'بہن',         # apa -> behan
            'دیدی': 'بہن',        # didi -> behan
        }
        
        # Compile all normalizations
        self.all_normalizations = {
            UrduDialect.PUNJABI_INFLUENCED: self.punjabi_normalizations,
            UrduDialect.SINDHI_INFLUENCED: self.sindhi_normalizations,
            UrduDialect.PASHTO_INFLUENCED: self.pashto_normalizations,
            UrduDialect.BALOCHI_INFLUENCED: self.balochi_normalizations,
            UrduDialect.DAKHINI: self.dakhini_normalizations,
        }
        
        # Patterns for dialect detection
        self.dialect_indicators = {
            UrduDialect.PUNJABI_INFLUENCED: [
                'ویلا', 'ویلے', 'ویلی', 'کیتا', 'کیتی', 'کیتے',
                'ہوئیا', 'گئیا', 'آئیا', 'نئیں', 'میں نوں', 'اوہ', 'ایہ'
            ],
            UrduDialect.SINDHI_INFLUENCED: [
                'ڪ', 'ڳ', 'ڄ', 'ڇ', 'ڏ', 'ڙ', 'ڻ',
                'پاڻي', 'ڪار', 'هن', 'هو', 'اهو', 'ڪيئن'
            ],
            UrduDialect.PASHTO_INFLUENCED: [
                'ښ', 'ځ', 'څ', 'ډ', 'ړ', 'ږ', 'ګ', 'ڼ',
                'اوبه', 'کور', 'ډوډۍ', 'زه', 'هغه', 'دا'
            ],
            UrduDialect.BALOCHI_INFLUENCED: [
                'شما', 'من', 'او', 'ای', 'آب', 'خانه', 'نان'
            ],
            UrduDialect.DAKHINI: [
                'کیکر', 'کیکو', 'کیدر', 'کیدو', 'نیر', 'آہار'
            ]
        }
    
    def detect_dialect(self, text: str) -> UrduDialect:
        """
        Detect the most likely dialect of the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Most likely dialect
        """
        if not text:
            return UrduDialect.UNKNOWN
        
        dialect_scores = {dialect: 0 for dialect in UrduDialect}
        
        # Count indicators for each dialect
        for dialect, indicators in self.dialect_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    dialect_scores[dialect] += text.count(indicator)
        
        # Find dialect with highest score
        max_score = max(dialect_scores.values())
        if max_score == 0:
            return UrduDialect.STANDARD
        
        for dialect, score in dialect_scores.items():
            if score == max_score:
                return dialect
        
        return UrduDialect.UNKNOWN
    
    def normalize_dialect(self, text: str, target_dialect: UrduDialect = UrduDialect.STANDARD) -> str:
        """
        Normalize dialectal variations to standard Urdu.
        
        Args:
            text: Input text to normalize
            target_dialect: Target dialect (default: standard)
            
        Returns:
            Normalized text
        """
        if not text or target_dialect != UrduDialect.STANDARD:
            return text
        
        normalized_text = text
        
        # Apply all dialect normalizations
        for dialect, normalizations in self.all_normalizations.items():
            for dialectal_form, standard_form in normalizations.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(dialectal_form) + r'\b'
                normalized_text = re.sub(pattern, standard_form, normalized_text)
        
        # Apply informal normalizations
        for informal_form, standard_form in self.informal_normalizations.items():
            pattern = r'\b' + re.escape(informal_form) + r'\b'
            normalized_text = re.sub(pattern, standard_form, normalized_text)
        
        return normalized_text
    
    def normalize_specific_dialect(self, text: str, dialect: UrduDialect) -> str:
        """
        Normalize text from a specific dialect to standard Urdu.
        
        Args:
            text: Input text to normalize
            dialect: Source dialect
            
        Returns:
            Normalized text
        """
        if not text or dialect == UrduDialect.STANDARD:
            return text
        
        normalized_text = text
        
        # Apply specific dialect normalizations
        if dialect in self.all_normalizations:
            normalizations = self.all_normalizations[dialect]
            for dialectal_form, standard_form in normalizations.items():
                pattern = r'\b' + re.escape(dialectal_form) + r'\b'
                normalized_text = re.sub(pattern, standard_form, normalized_text)
        
        # Always apply informal normalizations
        for informal_form, standard_form in self.informal_normalizations.items():
            pattern = r'\b' + re.escape(informal_form) + r'\b'
            normalized_text = re.sub(pattern, standard_form, normalized_text)
        
        return normalized_text
    
    def get_dialectal_features(self, text: str) -> Dict[str, List[str]]:
        """
        Extract dialectal features from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping dialects to found features
        """
        features = {dialect.value: [] for dialect in UrduDialect}
        
        if not text:
            return features
        
        for dialect, indicators in self.dialect_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    features[dialect.value].append(indicator)
        
        return features
    
    def get_normalization_suggestions(self, text: str) -> Dict[str, str]:
        """
        Get normalization suggestions for dialectal text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping original forms to suggested normalizations
        """
        suggestions = {}
        
        if not text:
            return suggestions
        
        # Check all normalizations
        all_normalizations = {}
        for dialect_normalizations in self.all_normalizations.values():
            all_normalizations.update(dialect_normalizations)
        all_normalizations.update(self.informal_normalizations)
        
        for dialectal_form, standard_form in all_normalizations.items():
            if dialectal_form in text and dialectal_form != standard_form:
                suggestions[dialectal_form] = standard_form
        
        return suggestions
    
    def standardize_text(self, text: str, detected_dialect: Optional[UrduDialect] = None) -> str:
        """
        Main function to standardize dialectal text.
        
        Args:
            text: Input text to standardize
            detected_dialect: Pre-detected dialect (optional)
            
        Returns:
            Standardized text
        """
        if not text:
            return text
        
        if detected_dialect is None:
            detected_dialect = self.detect_dialect(text)
        
        if detected_dialect == UrduDialect.STANDARD:
            # Still apply informal normalizations
            return self.normalize_dialect(text)
        else:
            return self.normalize_specific_dialect(text, detected_dialect)