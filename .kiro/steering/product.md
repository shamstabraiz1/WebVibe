# Product Overview

## Urdu Sentiment Analysis System

A comprehensive sentiment analysis system for Urdu text that handles dialectal variations, code-mixing, and script inconsistencies. Built with transformer models and designed for production use.

### Core Capabilities

- **Multi-dialect Support**: Handles 5 major Urdu dialects (Punjabi, Sindhi, Pashto, Balochi, Dakhini)
- **Script Normalization**: Converts Roman Urdu to Urdu script automatically with 500+ word mappings
- **Advanced Preprocessing**: Comprehensive text cleaning and normalization pipeline
- **Production API**: FastAPI-based REST service with batch processing support
- **High Performance**: GPU acceleration with <20ms prediction latency

### Target Use Cases

- Social media sentiment monitoring
- Customer feedback analysis
- News and content sentiment classification
- Research applications in Urdu NLP
- Multi-dialectal text processing

### Key Features

- Three-class sentiment classification (positive, negative, neutral)
- Confidence scoring and calibration
- Batch processing (up to 1000 texts)
- Health monitoring and metrics
- Comprehensive error handling
- CORS support for web integration