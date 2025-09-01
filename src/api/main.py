"""FastAPI application for Urdu sentiment analysis service."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import torch
import psutil
import os

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from ..models.urdu_sentiment_model import UrduSentimentModel
from ..models.data_models import (
    SentimentRequest, BatchSentimentRequest,
    SentimentResponse, BatchSentimentResponse,
    ErrorResponse, ModelInfo, HealthResponse
)
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model: Optional[UrduSentimentModel] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting Urdu Sentiment Analysis API")
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down Urdu Sentiment Analysis API")
    await cleanup_model()

async def load_model():
    """Load the sentiment analysis model."""
    global model
    try:
        # Check if pre-trained model exists
        model_path = Config.MODELS_DIR / "urdu_sentiment_model"
        
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            model = UrduSentimentModel.load_model(model_path)
        else:
            logger.info("Creating new model instance")
            model = UrduSentimentModel(
                model_name=Config.MODEL_NAME,
                num_classes=Config.NUM_CLASSES,
                max_length=Config.MAX_LENGTH,
                device=Config.get_device()
            )
        
        logger.info("Model loaded successfully")
        logger.info(f"Model info: {model.get_model_info()}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

async def cleanup_model():
    """Cleanup model resources."""
    global model
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model resources cleaned up")

def get_model() -> UrduSentimentModel:
    """Dependency to get the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    return model

# Create FastAPI app
app = FastAPI(
    title="Urdu Sentiment Analysis API",
    description="API for analyzing sentiment in Urdu text with dialectal variation support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred",
            details={"exception": str(exc)}
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).dict()
    )

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    try:
        # Check model status
        model_loaded = model is not None
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        
        # Get memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = {
            "cpu_percent": memory_info.percent,
            "cpu_available_gb": memory_info.available / (1024**3),
            "cpu_total_gb": memory_info.total / (1024**3)
        }
        
        if gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            memory_usage.update({
                "gpu_total_gb": gpu_memory / (1024**3),
                "gpu_allocated_gb": gpu_allocated / (1024**3),
                "gpu_percent": (gpu_allocated / gpu_memory) * 100
            })
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            memory_usage=memory_usage
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            gpu_available=False,
            memory_usage={}
        )

# Model info endpoint
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(current_model: UrduSentimentModel = Depends(get_model)):
    """Get model information."""
    try:
        model_info = current_model.get_model_info()
        
        return ModelInfo(
            model_name=model_info['model_name'],
            version="1.0.0",
            supported_languages=["urdu", "ur"],
            supported_dialects=list(model_info.get('supported_dialects', [])),
            max_text_length=model_info['max_length'],
            classes=["extremely_positive", "positive", "neutral", "negative", "extremely_negative"],
            performance_metrics=model_info.get('performance_metrics')
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model information"
        )

# Single text prediction endpoint
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(
    request: SentimentRequest,
    current_model: UrduSentimentModel = Depends(get_model)
):
    """Predict sentiment for a single text."""
    try:
        start_time = time.time()
        
        # Make prediction
        result = current_model.predict(
            text=request.text,
            return_confidence=request.include_confidence,
            return_processing_time=True,
            preprocess=True
        )
        
        # Create response
        response = SentimentResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            processing_time=result['processing_time'],
            dialect_detected=result.get('preprocessing_info', {}).get('detected_dialect'),
            confidence_scores=result['probabilities'] if request.include_confidence else None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(
    request: BatchSentimentRequest,
    current_model: UrduSentimentModel = Depends(get_model)
):
    """Predict sentiment for a batch of texts."""
    try:
        # Validate batch size
        if len(request.texts) > Config.get_api_config()['max_batch_size']:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum of {Config.get_api_config()['max_batch_size']}"
            )
        
        start_time = time.time()
        
        # Make batch predictions
        results = current_model.predict_batch(
            texts=request.texts,
            batch_size=Config.BATCH_SIZE,
            return_confidence=request.include_confidence,
            return_processing_time=True,
            preprocess=True
        )
        
        total_time = time.time() - start_time
        
        # Create response objects
        sentiment_responses = []
        for result in results:
            response = SentimentResponse(
                text=result['text'],
                sentiment=result['sentiment'],
                confidence=result['confidence'],
                processing_time=result['processing_time'],
                dialect_detected=result.get('preprocessing_info', {}).get('detected_dialect'),
                confidence_scores=result['probabilities'] if request.include_confidence else None
            )
            sentiment_responses.append(response)
        
        batch_response = BatchSentimentResponse(
            results=sentiment_responses,
            total_processing_time=total_time,
            batch_size=len(request.texts),
            average_processing_time=total_time / len(request.texts)
        )
        
        return batch_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Model management endpoints
@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (admin endpoint)."""
    try:
        background_tasks.add_task(reload_model_task)
        return {"message": "Model reload initiated"}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Model reload failed"
        )

async def reload_model_task():
    """Background task to reload the model."""
    global model
    try:
        logger.info("Reloading model...")
        await cleanup_model()
        await load_model()
        logger.info("Model reloaded successfully")
    except Exception as e:
        logger.error(f"Model reload task failed: {e}")

# Statistics endpoint
@app.get("/stats")
async def get_api_stats():
    """Get API statistics."""
    try:
        stats = {
            "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0,
            "model_loaded": model is not None,
            "gpu_available": torch.cuda.is_available(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "python_version": os.sys.version
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve statistics"
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Urdu Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info"
    }

# Record start time
start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    
    config = Config.get_api_config()
    uvicorn.run(
        "src.api.main:app",
        host=config['host'],
        port=config['port'],
        reload=False,  # Set to True for development
        workers=1  # Single worker for model sharing
    )