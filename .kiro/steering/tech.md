# Technology Stack

## Core Technologies

### Machine Learning & NLP
- **PyTorch**: Deep learning framework for model training and inference
- **Transformers (Hugging Face)**: Pre-trained BERT/XLM-RoBERTa models
- **scikit-learn**: Evaluation metrics and data splitting
- **datasets**: Dataset handling and processing

### API & Web Framework
- **FastAPI**: Modern async web framework for REST API
- **uvicorn**: ASGI server for production deployment
- **Pydantic**: Data validation and serialization
- **python-multipart**: File upload support

### Data Processing
- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical computations
- **aiofiles**: Async file operations

### Development & Testing
- **pytest**: Unit testing framework
- **pytest-asyncio**: Async testing support
- **tqdm**: Progress bars for training

### Visualization & Monitoring
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **psutil**: System monitoring

## Model Architecture

- **Base Models**: BERT multilingual, XLM-RoBERTa
- **Classification**: Custom sentiment classification head with dropout
- **Preprocessing**: Multi-stage text normalization pipeline
- **Device Support**: CUDA GPU acceleration with CPU fallback

## Common Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
python setup_environment.py
```

### Development
```bash
# Run API server
python run_api.py

# Run with development settings
python run_api.py --reload --log-level debug

# Run tests
pytest tests/

# Run specific test module
pytest tests/test_urdu_bert_model.py -v
```

### Training
```bash
# Train model with default settings
python train_model.py --data_path data/sentiment_data.csv

# Train with custom parameters
python train_model.py --epochs 5 --batch_size 32 --learning_rate 1e-5
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "یہ بہت اچھا ہے"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["یہ اچھا ہے", "برا ہے"]}'
```

## Configuration

- **Main Config**: `config/config.py` - Central configuration management
- **Model Settings**: BERT model name, max length, batch size
- **API Settings**: Host, port, CORS, timeout settings
- **Training**: Learning rate, epochs, validation split
- **GPU**: CUDA settings and memory management