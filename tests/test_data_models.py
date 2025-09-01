"""Tests for data models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models.data_models import (
    SentimentRequest, BatchSentimentRequest, SentimentResponse,
    BatchSentimentResponse, UrduSentimentSample, ErrorResponse,
    ModelInfo, HealthResponse, SentimentLabel, DialectHint
)

class TestSentimentRequest:
    """Test cases for SentimentRequest model."""
    
    def test_valid_request(self):
        """Test valid sentiment request."""
        request = SentimentRequest(
            text="یہ بہت اچھا ہے",
            include_confidence=True,
            dialect_hint=DialectHint.STANDARD
        )
        assert request.text == "یہ بہت اچھا ہے"
        assert request.include_confidence is True
        assert request.dialect_hint == DialectHint.STANDARD
    
    def test_minimal_request(self):
        """Test minimal valid request."""
        request = SentimentRequest(text="اچھا")
        assert request.text == "اچھا"
        assert request.include_confidence is True  # default
        assert request.dialect_hint is None  # default
    
    def test_empty_text_validation(self):
        """Test validation for empty text."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentRequest(text="")
        assert "Text cannot be empty" in str(exc_info.value)
    
    def test_whitespace_only_text_validation(self):
        """Test validation for whitespace-only text."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentRequest(text="   ")
        assert "Text cannot be empty" in str(exc_info.value)
    
    def test_too_short_text_validation(self):
        """Test validation for too short text."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentRequest(text="a")
        assert "must contain at least 2 characters" in str(exc_info.value)
    
    def test_text_stripping(self):
        """Test that text is properly stripped."""
        request = SentimentRequest(text="  یہ اچھا ہے  ")
        assert request.text == "یہ اچھا ہے"

class TestBatchSentimentRequest:
    """Test cases for BatchSentimentRequest model."""
    
    def test_valid_batch_request(self):
        """Test valid batch request."""
        request = BatchSentimentRequest(
            texts=["یہ اچھا ہے", "یہ برا ہے"],
            include_confidence=True,
            dialect_hints=[DialectHint.STANDARD, DialectHint.PUNJABI_INFLUENCED]
        )
        assert len(request.texts) == 2
        assert request.texts[0] == "یہ اچھا ہے"
        assert len(request.dialect_hints) == 2
    
    def test_minimal_batch_request(self):
        """Test minimal valid batch request."""
        request = BatchSentimentRequest(texts=["اچھا", "برا"])
        assert len(request.texts) == 2
        assert request.include_confidence is True
        assert request.dialect_hints is None
    
    def test_empty_texts_validation(self):
        """Test validation for empty texts list."""
        with pytest.raises(ValidationError) as exc_info:
            BatchSentimentRequest(texts=[])
        assert "Texts list cannot be empty" in str(exc_info.value)
    
    def test_empty_text_in_list_validation(self):
        """Test validation for empty text in list."""
        with pytest.raises(ValidationError) as exc_info:
            BatchSentimentRequest(texts=["اچھا", "", "برا"])
        assert "Text at index 1 cannot be empty" in str(exc_info.value)
    
    def test_dialect_hints_length_mismatch(self):
        """Test validation for dialect hints length mismatch."""
        with pytest.raises(ValidationError) as exc_info:
            BatchSentimentRequest(
                texts=["اچھا", "برا"],
                dialect_hints=[DialectHint.STANDARD]  # Only one hint for two texts
            )
        assert "dialect_hints length must match texts length" in str(exc_info.value)
    
    def test_max_batch_size_validation(self):
        """Test validation for maximum batch size."""
        large_texts = ["اچھا"] * 1001  # Exceeds max of 1000
        with pytest.raises(ValidationError) as exc_info:
            BatchSentimentRequest(texts=large_texts)
        assert "ensure this value has at most 1000 items" in str(exc_info.value)

class TestSentimentResponse:
    """Test cases for SentimentResponse model."""
    
    def test_valid_response(self):
        """Test valid sentiment response."""
        response = SentimentResponse(
            text="یہ اچھا ہے",
            sentiment=SentimentLabel.POSITIVE,
            confidence=0.95,
            processing_time=0.1,
            dialect_detected=DialectHint.STANDARD,
            confidence_scores={
                "positive": 0.95,
                "negative": 0.03,
                "neutral": 0.02
            }
        )
        assert response.sentiment == SentimentLabel.POSITIVE
        assert response.confidence == 0.95
        assert response.confidence_scores["positive"] == 0.95
    
    def test_confidence_range_validation(self):
        """Test confidence score range validation."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentResponse(
                text="test",
                sentiment=SentimentLabel.POSITIVE,
                confidence=1.5,  # Invalid: > 1.0
                processing_time=0.1
            )
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)
    
    def test_negative_processing_time_validation(self):
        """Test negative processing time validation."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentResponse(
                text="test",
                sentiment=SentimentLabel.POSITIVE,
                confidence=0.9,
                processing_time=-0.1  # Invalid: negative
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

class TestUrduSentimentSample:
    """Test cases for UrduSentimentSample model."""
    
    def test_valid_sample(self):
        """Test valid training sample."""
        sample = UrduSentimentSample(
            text="یہ فلم بہت اچھی ہے",
            label=SentimentLabel.POSITIVE,
            dialect=DialectHint.STANDARD,
            source="movie_reviews",
            confidence=1.0,
            metadata={"reviewer_id": "123", "rating": 5}
        )
        assert sample.text == "یہ فلم بہت اچھی ہے"
        assert sample.label == SentimentLabel.POSITIVE
        assert sample.metadata["rating"] == 5
    
    def test_minimal_sample(self):
        """Test minimal valid sample."""
        sample = UrduSentimentSample(
            text="اچھا",
            label=SentimentLabel.POSITIVE,
            source="test"
        )
        assert sample.dialect == DialectHint.UNKNOWN  # default
        assert sample.confidence == 1.0  # default
        assert sample.metadata is None  # default
    
    def test_empty_text_validation(self):
        """Test validation for empty training text."""
        with pytest.raises(ValidationError) as exc_info:
            UrduSentimentSample(
                text="",
                label=SentimentLabel.POSITIVE,
                source="test"
            )
        assert "Training text cannot be empty" in str(exc_info.value)

class TestErrorResponse:
    """Test cases for ErrorResponse model."""
    
    def test_valid_error_response(self):
        """Test valid error response."""
        error = ErrorResponse(
            error_code="INVALID_INPUT",
            message="Invalid input provided",
            details={"field": "text", "issue": "too_short"}
        )
        assert error.error_code == "INVALID_INPUT"
        assert error.details["field"] == "text"
        assert isinstance(error.timestamp, datetime)
    
    def test_minimal_error_response(self):
        """Test minimal error response."""
        error = ErrorResponse(
            error_code="UNKNOWN_ERROR",
            message="An unknown error occurred"
        )
        assert error.details is None
        assert isinstance(error.timestamp, datetime)

class TestModelInfo:
    """Test cases for ModelInfo model."""
    
    def test_valid_model_info(self):
        """Test valid model info."""
        info = ModelInfo(
            model_name="urdu-sentiment-bert",
            version="1.0.0",
            supported_languages=["urdu", "ur"],
            supported_dialects=[DialectHint.STANDARD, DialectHint.PUNJABI_INFLUENCED],
            max_text_length=512,
            classes=[SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL],
            performance_metrics={"accuracy": 0.87, "f1_score": 0.85}
        )
        assert info.model_name == "urdu-sentiment-bert"
        assert len(info.supported_dialects) == 2
        assert info.performance_metrics["accuracy"] == 0.87

class TestHealthResponse:
    """Test cases for HealthResponse model."""
    
    def test_valid_health_response(self):
        """Test valid health response."""
        health = HealthResponse(
            status="healthy",
            model_loaded=True,
            gpu_available=True,
            memory_usage={"cpu_percent": 45.2, "gpu_percent": 23.1}
        )
        assert health.status == "healthy"
        assert health.model_loaded is True
        assert health.memory_usage["cpu_percent"] == 45.2
        assert isinstance(health.timestamp, datetime)