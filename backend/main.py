"""
BitScan - Main FastAPI application for Bitcoin fraud detection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to Python path for imports (works in both dev and production)
from pathlib import Path
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import our modules
from api.routes import router as api_router
from api.timeseries import router as ts_router
from blockchain.analyzer import BlockchainAnalyzer
from ml.enhanced_fraud_detector import EnhancedFraudDetector
from ml.fraud_detector import FraudDetector

app = FastAPI(
    title="BitScan - Bitcoin Scam Pattern Analyzer",
    description="Detect fraudulent Bitcoin investment schemes through blockchain analytics and ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
from fastapi.middleware.cors import CORSMiddleware

# List all allowed frontend URLs
allowed_origins = [
    "https://test-bitscan.vercel.app",
    "https://www.test-bitscan.vercel.app",  # in case you use www
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,   # exact match only
    allow_credentials=True,          # needed if sending cookies or auth headers
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # explicit methods
    allow_headers=["Authorization", "Content-Type"],            # explicit headers
)


# Include API routes BEFORE static file mounts
app.include_router(api_router, prefix="/api/v1", tags=["BitScan API"])
app.include_router(ts_router, prefix="/api/v1", tags=["Wallet Time Series"])

# Mount static files with caching
if os.path.exists("frontend/dist"):
    # Production: serve React build
    from fastapi.staticfiles import StaticFiles
    
    class CachedStaticFiles(StaticFiles):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "public, max-age=3600"  # 1 hour cache
            return response
    
    app.mount("/static", CachedStaticFiles(directory="frontend/dist/assets"), name="static")

# Add error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Global instances
blockchain_analyzer = None
fraud_detector = None
legacy_fraud_detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup with enhanced fraud detector"""
    global blockchain_analyzer, fraud_detector, legacy_fraud_detector
    
    print("🚀 Initializing BitScan services...")
    
    # Initialize blockchain analyzer
    blockchain_analyzer = BlockchainAnalyzer()
    
    # Initialize enhanced fraud detector
    try:
        fraud_detector = EnhancedFraudDetector()
        print("✅ Enhanced fraud detector initialized successfully!")
    except Exception as e:
        print(f"⚠️ Enhanced fraud detector failed to initialize: {e}")
        print("📋 Falling back to legacy fraud detector...")
        fraud_detector = FraudDetector()
    
    # Keep legacy detector as backup
    legacy_fraud_detector = FraudDetector()
    
    print("✅ BitScan services initialized successfully!")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the React app or fallback HTML"""
    # Check if React build exists
    react_index = "frontend/dist/index.html"
    if os.path.exists(react_index):
        with open(react_index, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Minimal fallback for development
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BitScan - Bitcoin Fraud Detection</title>
        <style>
            body { font-family: system-ui, sans-serif; margin: 0; padding: 40px; background: #f5f7fa; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); text-align: center; }
            .btn { background: #1976d2; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; display: inline-block; margin: 10px; transition: all 0.2s; }
            .btn:hover { background: #1565c0; transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 BitScan</h1>
            <p>Bitcoin Fraud Detection System</p>
            <a href="http://localhost:5173" class="btn">React Development UI</a>
            <a href="/docs" class="btn">API Documentation</a>
        </div>
    </body>
    </html>
    """

# Mount React SPA at the end to serve as fallback
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="react-app")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BitScan"}

# Handle Chrome DevTools requests to reduce 404 logs
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Handle Chrome DevTools requests"""
    from fastapi.responses import Response
    return Response(status_code=404, content="", media_type="application/json")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
