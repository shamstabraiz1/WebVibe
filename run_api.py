"""Script to run the Urdu Sentiment Analysis API."""

import uvicorn
import argparse
import logging
from config.config import Config

def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description="Run Urdu Sentiment Analysis API")
    parser.add_argument("--host", type=str, default=Config.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=Config.API_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    print("🚀 Starting Urdu Sentiment Analysis API")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"🔍 Health Check: http://localhost:8000/health")
    
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()