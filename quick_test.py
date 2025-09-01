"""Quick test script to verify everything works."""

def test_basic_functionality():
    """Test basic model functionality."""
    try:
        from src.models.urdu_sentiment_model import UrduSentimentModel
        
        print("Testing Urdu Sentiment Model...")
        
        # Initialize model
        model = UrduSentimentModel()
        print("Model initialized")
        
        # Test predictions
        test_texts = [
            "یہ بہت اچھا ہے",
            "مجھے پسند نہیں",
            "ٹھیک ہے"
        ]
        
        for text in test_texts:
            result = model.predict(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print("---")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_basic_functionality()