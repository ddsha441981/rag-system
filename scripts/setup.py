#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/uploads", "data/processed", "data/models", "data/cache",
        "logs", "static/css", "static/js", "static/images"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_requirements():
    """Install Python requirements"""
    print("📦 Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def download_models():
    """Download required language models"""
    print("🤖 Downloading language models...")

    # Download spaCy model
    try:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ Downloaded spaCy English model")
    except Exception as e:
        print(f"❌ Failed to download spaCy model: {e}")

    # Download NLTK data
    try:
        subprocess.run([sys.executable, "-c", "import nltk; nltk.download('punkt')"])
        print("✅ Downloaded NLTK punkt tokenizer")
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")


def create_env_file():
    """Create .env file from template"""
    if not os.path.exists(".env"):
        print("⚙️ Creating .env file...")
        env_content = """# API Keys
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
CHROMA_PERSIST_DIR=./data/chroma_db
COLLECTION_NAME=enhanced_knowledge_base

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL_HOURS=24

# Model Configuration
DEFAULT_LLM_MODEL=llama3-8b-8192
EMBEDDING_MODEL=all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Search Configuration
DEFAULT_SEARCH_K=5
HYBRID_SEARCH_ALPHA=0.7
MAX_CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Application Settings
MAX_UPLOAD_SIZE_MB=50
ALLOWED_FILE_TYPES=pdf,docx,txt,md,csv,xlsx,html,json
LOG_LEVEL=INFO
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("⚠️  Please update .env with your actual API keys!")
    else:
        print("✅ .env file already exists")


def setup_project():
    """Main setup function"""
    print("🚀 Setting up Enhanced RAG System...")

    create_directories()
    install_requirements()
    download_models()
    create_env_file()

    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Update .env file with your API keys")
    print("2. Start Redis: redis-server")
    print("3. Run API: python app/main.py")
    print("4. Run Streamlit: streamlit run interfaces/streamlit_app.py")


if __name__ == "__main__":
    setup_project()
