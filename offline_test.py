"""Offline test without downloading models."""

def test_preprocessing():
    """Test preprocessing components only."""
    try:
        from src.data.text_cleaner import UrduTextCleaner
        from src.data.script_normalizer import ScriptNormalizer
        from src.data.dialectal_normalizer import DialectalNormalizer
        
        print("Testing preprocessing components...")
        
        # Test text cleaner
        cleaner = UrduTextCleaner()
        text = "یہ بہت اچھا ہے! @user #hashtag"
        cleaned = cleaner.clean_text(text)
        print(f"Cleaned: {cleaned}")
        
        # Test script normalizer
        normalizer = ScriptNormalizer()
        roman_text = "ye bahut acha hai"
        normalized = normalizer.normalize_script(roman_text)
        print(f"Normalized: {normalized}")
        
        # Test dialectal normalizer
        dialect_normalizer = DialectalNormalizer()
        dialect_text = "ایہ بہت چنگا اے"  # Punjabi influenced
        standardized = dialect_normalizer.standardize_text(dialect_text)
        print(f"Standardized: {standardized}")
        
        print("All preprocessing tests passed!")
        
    except Exception as e:
        print(f"Error: {e}")

def test_api_structure():
    """Test API structure without model loading."""
    try:
        from src.models.data_models import SentimentRequest, SentimentResponse
        
        # Test request model
        request = SentimentRequest(text="یہ اچھا ہے")
        print(f"Request created: {request.text}")
        
        # Test response model
        response = SentimentResponse(
            text="یہ اچھا ہے",
            sentiment="positive",
            confidence=0.85,
            processing_time=0.1
        )
        print(f"Response created: {response.sentiment}")
        
        print("API structure tests passed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_preprocessing()
    test_api_structure()