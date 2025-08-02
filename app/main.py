from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from contextlib import asynccontextmanager
from interfaces.fastapi_routes import router
from app.config import settings

os.makedirs("logs", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper() if hasattr(settings, 'log_level') else 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events
    logger.info("🚀 Starting Enhanced RAG System")
    logger.info("✅ System initialized successfully")
    yield
    # Shutdown events
    logger.info("🛑 Shutting down Enhanced RAG System")

app = FastAPI(
    title="Enhanced RAG System",
    description="Advanced Retrieval-Augmented Generation with multi-format support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Enhanced RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2025-08-01T21:30:00Z",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

