# Project Structure

## Directory Organization

```
urdu-sentiment-analysis/
├── config/                 # Configuration management
│   └── config.py          # Central configuration class
├── src/                   # Main source code
│   ├── api/              # FastAPI web service
│   │   └── main.py       # API endpoints and middleware
│   ├── data/             # Data processing utilities
│   │   ├── text_cleaner.py        # Text cleaning and normalization
│   │   ├── script_normalizer.py   # Roman to Urdu script conversion
│   │   └── dialectal_normalizer.py # Dialect standardization
│   ├── models/           # ML models and data structures
│   │   ├── data_models.py         # Pydantic models for API
│   │   ├── urdu_bert_model.py     # BERT wrapper for Urdu
│   │   ├── classification_head.py  # Sentiment classification layer
│   │   └── urdu_sentiment_model.py # Complete integrated model
│   └── evaluation/       # Model evaluation tools
├── tests/                # Unit and integration tests
├── .kiro/               # Kiro configuration
│   └── steering/        # AI assistant guidance
└── [root files]        # Entry points and configuration
```

## Architecture Patterns

### Modular Design
- **Separation of Concerns**: Each module handles a specific aspect (preprocessing, modeling, API)
- **Dependency Injection**: Models accept configuration objects for flexibility
- **Interface Consistency**: All components follow similar initialization and usage patterns

### Data Flow
1. **Input**: Raw Urdu text (mixed scripts, dialectal variations)
2. **Preprocessing**: Text cleaning → Script normalization → Dialect standardization
3. **Tokenization**: BERT tokenizer with Urdu-specific handling
4. **Model**: BERT encoder → Classification head → Sentiment prediction
5. **Output**: Structured response with confidence scores

### Configuration Management
- **Centralized Config**: Single `Config` class in `config/config.py`
- **Environment Variables**: Support for deployment-specific settings
- **Type Safety**: Pydantic models for request/response validation

## Key Components

### Data Processing Pipeline (`src/data/`)
- **UrduTextCleaner**: Removes URLs, mentions, normalizes punctuation
- **ScriptNormalizer**: 500+ Roman-to-Urdu word mappings
- **DialectalNormalizer**: Handles 5 major Urdu dialects
- **TextPreprocessor**: Orchestrates the complete pipeline

### Model Architecture (`src/models/`)
- **UrduBertModel**: Wrapper around Hugging Face transformers
- **SentimentClassificationHead**: Custom classification layer with dropout
- **UrduSentimentModel**: Complete end-to-end model
- **DataModels**: Pydantic schemas for type safety

### API Service (`src/api/`)
- **FastAPI Application**: Async REST API with OpenAPI docs
- **Health Monitoring**: System metrics and model status
- **Batch Processing**: Efficient handling of multiple texts
- **Error Handling**: Structured error responses

## Naming Conventions

### Files and Modules
- **Snake Case**: `urdu_bert_model.py`, `text_cleaner.py`
- **Descriptive Names**: Clear indication of functionality
- **Consistent Suffixes**: `_model.py`, `_normalizer.py`, `_cleaner.py`

### Classes and Functions
- **PascalCase Classes**: `UrduSentimentModel`, `TextPreprocessor`
- **Snake Case Functions**: `predict_sentiment()`, `preprocess_text()`
- **Verb-Noun Pattern**: Functions start with action verbs

### Constants and Configuration
- **UPPER_CASE**: Configuration constants in `Config` class
- **Grouped by Purpose**: Model settings, API settings, training settings

## Testing Structure

### Test Organization
- **Mirror Source Structure**: Tests follow `src/` directory layout
- **Descriptive Names**: `test_urdu_bert_model.py`, `test_text_cleaner.py`
- **Comprehensive Coverage**: Unit tests for all major components

### Test Patterns
- **Fixture-Based**: Reusable test data and model instances
- **Async Testing**: Support for FastAPI endpoint testing
- **Mock External Dependencies**: Isolate unit tests from external services

## Entry Points

### Main Scripts
- **`run_api.py`**: Start the FastAPI server with configuration options
- **`train_model.py`**: Train sentiment models with custom datasets
- **`setup_environment.py`**: Initialize project directories and dependencies

### Import Patterns
- **Relative Imports**: Within packages use relative imports
- **Absolute Imports**: From root use `from src.models import ...`
- **Config Import**: `from config.config import Config`