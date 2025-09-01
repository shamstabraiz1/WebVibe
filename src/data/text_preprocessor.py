"""Integrated text preprocessing pipeline for Urdu sentiment analysis."""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .text_cleaner import UrduTextCleaner
from .script_normalizer import ScriptNormalizer
from .dialectal_normalizer import DialectalNormalizer, UrduDialect

class PreprocessingLevel(Enum):
    """Preprocessing intensity levels."""
    MINIMAL = "minimal"      # Basic cleaning only
    STANDARD = "standard"    # Standard cleaning + normalization
    AGGRESSIVE = "aggressive" # All preprocessing steps

@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    
    # Text cleaning options
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = False  # Keep hashtag content
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    normalize_punctuation: bool = True
    normalize_repeated_chars: bool = True
    normalize_whitespace: bool = True
    remove_special_chars: bool = True
    normalize_unicode: bool = True
    convert_digits: bool = True
    fix_encoding: bool = True
    keep_punctuation: bool = True
    
    # Script normalization options
    normalize_script: bool = True
    force_script_conversion: bool = False
    
    # Dialectal normalization options
    normalize_dialects: bool = True
    target_dialect: UrduDialect = UrduDialect.STANDARD
    auto_detect_dialect: bool = True
    
    # Performance options
    enable_caching: bool = True
    cache_size: int = 1000
    
    @classmethod
    def minimal(cls) -> 'PreprocessingConfig':
        """Create minimal preprocessing configuration."""
        return cls(
            remove_urls=True,
            remove_mentions=True,
            remove_emails=True,
            normalize_whitespace=True,
            normalize_script=False,
            normalize_dialects=False,
            enable_caching=False
        )
    
    @classmethod
    def standard(cls) -> 'PreprocessingConfig':
        """Create standard preprocessing configuration."""
        return cls()  # Uses default values
    
    @classmethod
    def aggressive(cls) -> 'PreprocessingConfig':
        """Create aggressive preprocessing configuration."""
        return cls(
            remove_hashtags=True,
            remove_special_chars=True,
            force_script_conversion=True,
            normalize_dialects=True,
            enable_caching=True
        )

@dataclass
class PreprocessingResult:
    """Result of text preprocessing."""
    
    original_text: str
    processed_text: str
    processing_time: float
    
    # Detected characteristics
    original_script_type: str
    detected_dialect: UrduDialect
    
    # Applied transformations
    transformations_applied: List[str]
    
    # Statistics
    original_stats: Dict[str, Any]
    processed_stats: Dict[str, Any]
    
    # Suggestions and warnings
    suggestions: List[str]
    warnings: List[str]

class TextPreprocessor:
    """Integrated text preprocessing pipeline for Urdu text."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the text preprocessor.
        
        Args:
            config: Preprocessing configuration (uses standard if None)
        """
        self.config = config or PreprocessingConfig.standard()
        
        # Initialize component processors
        self.text_cleaner = UrduTextCleaner()
        self.script_normalizer = ScriptNormalizer()
        self.dialectal_normalizer = DialectalNormalizer()
        
        # Initialize cache if enabled
        self.cache = {} if self.config.enable_caching else None
        self.cache_hits = 0
        self.cache_misses = 0
    
    def preprocess(self, text: str) -> PreprocessingResult:
        """
        Main preprocessing function.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            PreprocessingResult with processed text and metadata
        """
        start_time = time.time()
        
        # Initialize result
        original_text = text
        processed_text = text
        transformations_applied = []
        suggestions = []
        warnings = []
        
        # Handle empty or invalid input
        if not text or not isinstance(text, str):
            warnings.append("Empty or invalid input text")
            return PreprocessingResult(
                original_text=original_text,
                processed_text="",
                processing_time=time.time() - start_time,
                original_script_type="unknown",
                detected_dialect=UrduDialect.UNKNOWN,
                transformations_applied=transformations_applied,
                original_stats={},
                processed_stats={},
                suggestions=suggestions,
                warnings=warnings
            )
        
        # Get original statistics
        original_stats = self.text_cleaner.get_text_stats(text)
        original_script_type = self.script_normalizer.detect_script_type(text)
        detected_dialect = UrduDialect.UNKNOWN
        
        # Step 1: Text Cleaning
        if any([
            self.config.remove_urls, self.config.remove_mentions,
            self.config.remove_hashtags, self.config.remove_emails,
            self.config.remove_phone_numbers, self.config.normalize_punctuation,
            self.config.normalize_repeated_chars, self.config.normalize_whitespace,
            self.config.remove_special_chars, self.config.normalize_unicode,
            self.config.convert_digits, self.config.fix_encoding
        ]):
            processed_text = self.text_cleaner.clean_text(
                processed_text,
                remove_urls=self.config.remove_urls,
                remove_mentions=self.config.remove_mentions,
                remove_hashtags=self.config.remove_hashtags,
                remove_emails=self.config.remove_emails,
                remove_phone_numbers=self.config.remove_phone_numbers,
                normalize_punctuation=self.config.normalize_punctuation,
                normalize_repeated_chars=self.config.normalize_repeated_chars,
                normalize_whitespace=self.config.normalize_whitespace,
                remove_special_chars=self.config.remove_special_chars,
                normalize_unicode=self.config.normalize_unicode,
                convert_digits=self.config.convert_digits,
                fix_encoding=self.config.fix_encoding,
                keep_punctuation=self.config.keep_punctuation
            )
            transformations_applied.append("text_cleaning")
        
        # Step 2: Script Normalization
        if self.config.normalize_script:
            script_before = processed_text
            processed_text = self.script_normalizer.normalize_script(
                processed_text,
                force_conversion=self.config.force_script_conversion
            )
            if script_before != processed_text:
                transformations_applied.append("script_normalization")
                
                # Add suggestions based on script type
                if original_script_type == "roman":
                    suggestions.append("Roman Urdu text was converted to Urdu script")
                elif original_script_type == "mixed":
                    suggestions.append("Mixed script text was normalized")
        
        # Step 3: Dialectal Normalization
        if self.config.normalize_dialects:
            dialect_before = processed_text
            
            if self.config.auto_detect_dialect:
                detected_dialect = self.dialectal_normalizer.detect_dialect(processed_text)
                processed_text = self.dialectal_normalizer.standardize_text(
                    processed_text, detected_dialect
                )
            else:
                processed_text = self.dialectal_normalizer.normalize_dialect(
                    processed_text, self.config.target_dialect
                )
            
            if dialect_before != processed_text:
                transformations_applied.append("dialectal_normalization")
                
                # Add suggestions based on detected dialect
                if detected_dialect != UrduDialect.STANDARD:
                    suggestions.append(f"Detected {detected_dialect.value} dialect features")
        
        # Get processed statistics
        processed_stats = self.text_cleaner.get_text_stats(processed_text)
        
        # Create result
        result = PreprocessingResult(
            original_text=original_text,
            processed_text=processed_text,
            processing_time=time.time() - start_time,
            original_script_type=original_script_type,
            detected_dialect=detected_dialect,
            transformations_applied=transformations_applied,
            original_stats=original_stats,
            processed_stats=processed_stats,
            suggestions=suggestions,
            warnings=warnings
        )
        
        return result
    
    def preprocess_batch(self, texts: List[str]) -> List[PreprocessingResult]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of PreprocessingResult objects
        """
        return [self.preprocess(text) for text in texts]
    
    def update_config(self, new_config: PreprocessingConfig):
        """
        Update preprocessing configuration.
        
        Args:
            new_config: New configuration to use
        """
        self.config = new_config