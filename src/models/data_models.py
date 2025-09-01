"""Data models for Urdu Sentiment Analysis API and training."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    EXTREMELY_POSITIVE = "extremely_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    EXTREMELY_NEGATIVE = "extremely_negative"

class DialectHint(str, Enum):
    """Supported Urdu dialect hints."""
    STANDARD = "standard"
    PUNJABI_INFLUENCED = "punjabi_influenced"
    SINDHI_INFLUENCED = "sindhi_influenced"
    PASHTO_INFLUENCED = "pashto_influenced"
    BALOCHI_INFLUENCED = "balochi_influenced"
    UNKNOWN = "unknown"

class SentimentRequest(BaseModel):
    """Request model for single text sentiment analysis."""
    
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="Urdu text to analyze for sentiment"
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores in response"
    )
    dialect_hint: Optional[DialectHint] = Field(
        default=None,
        description="Optional hint about the dialect of the input text"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate input text."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        
        # Check for minimum meaningful content
        if len(v.strip()) < 2:
            raise ValueError("Text must contain at least 2 characters")
        
        return v.strip()

class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis."""
    
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of Urdu texts to analyze"
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores in responses"
    )
    dialect_hints: Optional[List[Optional[DialectHint]]] = Field(
        default=None,
        description="Optional dialect hints for each text (must match texts length)"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate input texts."""
        if not v:
            raise ValueError("Texts list cannot be empty")
        
        validated_texts = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or only whitespace")
            if len(text.strip()) < 2:
                raise ValueError(f"Text at index {i} must contain at least 2 characters")
            validated_texts.append(text.strip())
        
        return validated_texts
    
    @validator('dialect_hints')
    def validate_dialect_hints(cls, v, values):
        """Validate dialect hints match texts length."""
        if v is not None and 'texts' in values:
            if len(v) != len(values['texts']):
                raise ValueError("dialect_hints length must match texts length")
        return v

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    
    text: str = Field(..., description="Original input text")
    sentiment: SentimentLabel = Field(..., description="Predicted sentiment label")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score for the prediction"
    )
    processing_time: float = Field(
        ..., 
        ge=0.0,
        description="Processing time in seconds"
    )
    dialect_detected: Optional[DialectHint] = Field(
        default=None,
        description="Detected dialect if available"
    )
    confidence_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Confidence scores for all classes"
    )

class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis."""
    
    results: List[SentimentResponse] = Field(
        ...,
        description="List of sentiment analysis results"
    )
    total_processing_time: float = Field(
        ...,
        ge=0.0,
        description="Total processing time for the batch in seconds"
    )
    batch_size: int = Field(
        ...,
        ge=1,
        description="Number of texts processed"
    )
    average_processing_time: float = Field(
        ...,
        ge=0.0,
        description="Average processing time per text in seconds"
    )

class UrduSentimentSample(BaseModel):
    """Training data sample model."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Urdu text content"
    )
    label: SentimentLabel = Field(
        ...,
        description="Ground truth sentiment label"
    )
    dialect: Optional[DialectHint] = Field(
        default=DialectHint.UNKNOWN,
        description="Dialect information if available"
    )
    source: str = Field(
        ...,
        description="Source of the data (e.g., 'twitter', 'news', 'reviews')"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Annotation confidence (1.0 for high confidence)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the sample"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate training text."""
        if not v or not v.strip():
            raise ValueError("Training text cannot be empty")
        return v.strip()

class ErrorResponse(BaseModel):
    """Error response model."""
    
    error_code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )

class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    supported_languages: List[str] = Field(..., description="Supported languages")
    supported_dialects: List[DialectHint] = Field(..., description="Supported dialects")
    max_text_length: int = Field(..., description="Maximum supported text length")
    classes: List[SentimentLabel] = Field(..., description="Supported sentiment classes")
    performance_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Model performance metrics"
    )

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Health check timestamp"
    )
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Optional[Dict[str, float]] = Field(
        default=None,
        description="Memory usage information"
    )