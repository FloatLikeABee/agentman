#!/bin/bash

# Ground Control Startup Script

echo "🚀 Starting Ground Control with MCP Support..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check LLM provider configuration
echo "🔍 Checking LLM provider configuration..."
if [ -z "$GEMINI_API_KEY" ] && [ -z "$QWEN_API_KEY" ] && [ -z "$GLM_API_KEY" ]; then
    echo "⚠️  No LLM API keys configured. Please set GEMINI_API_KEY, QWEN_API_KEY, or GLM_API_KEY in your .env file."
    echo "   You can still run the system, but agent functionality will be limited."
fi

# Install Python dependencies if not already installed
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install frontend dependencies if not already installed
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Database settings
CHROMA_PERSIST_DIRECTORY=./chroma_db

# LLM Provider settings
DEFAULT_LLM_PROVIDER=gemini

# Gemini settings
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_DEFAULT_MODEL=gemini-2.5-flash

# Qwen settings
QWEN_API_KEY=your_qwen_api_key_here
QWEN_DEFAULT_MODEL=qwen3-max

# GLM settings
GLM_API_KEY=your_glm_api_key_here
GLM_DEFAULT_MODEL=glm-4.6

# Embedding settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# CORS settings
CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]

# Email settings (optional)
SMTP_SERVER=
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=

# Financial API settings (optional)
ALPHA_VANTAGE_API_KEY=
EOF
    echo "✅ Created .env file. Please edit it with your configuration."
fi

# Start the backend
echo "🔧 Starting backend server..."
python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start the frontend
echo "🎨 Starting frontend..."
cd frontend
npm start &
FRONTEND_PID=$!

echo "✅ Ground Control is starting up!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the system"

# Wait for user to stop
trap "echo '🛑 Stopping Ground Control...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait 