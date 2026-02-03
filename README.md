# Ground Control with MCP Support

A comprehensive Retrieval-Augmented Generation (RAG) system with Model Context Protocol (MCP) support for Ollama models. This system provides a complete solution for building, managing, and deploying AI agents with RAG capabilities.

## Features

### 🧠 Ground Control
- **Vector Database**: ChromaDB integration for efficient document storage and retrieval
- **Multiple Formats**: Support for JSON, CSV, and text data formats
- **Data Validation**: Built-in validation for all data inputs
- **Collection Management**: Create, query, and manage RAG collections
- **Semantic Search**: Advanced embedding-based search capabilities

### 🤖 Agent Management
- **Configurable Agents**: Create agents with different types (RAG, Tool, Hybrid)
- **Model Support**: Integration with any Ollama model
- **Tool Integration**: Connect agents with various tools
- **Real-time Execution**: Run agents and get responses through the API

### 🛠️ Tool System
- **Email Tools**: Send emails with SMTP configuration
- **Web Search**: DuckDuckGo integration for web searches
- **Calculator**: Mathematical computation capabilities
- **Financial Data**: Real-time stock prices and financial data (yfinance/Yahoo Finance - most reliable, free, no API key required)
- **Wikipedia**: Wikipedia search integration
- **Browser Automation**: AI-powered browser control using LangChain + Playwright for automated web tasks
- **Custom Tools**: Extensible tool system for custom functionality

### 🔌 MCP (Model Context Protocol)
- **Stable Communication**: Reliable protocol for AI model communication
- **Real-time Updates**: Live status monitoring and updates
- **Client Support**: Multiple client connections
- **Message Broadcasting**: Broadcast messages to all connected clients

### 🎨 Modern UI
- **React + Material-UI**: Beautiful, responsive web interface
- **Real-time Updates**: Live system status and monitoring
- **Intuitive Design**: Easy-to-use interface for all operations
- **Mobile Responsive**: Works on all device sizes

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │   FastAPI Backend│    │   Ollama Models │
│                 │◄──►│                 │◄──►│                 │
│  Material-UI    │    │  Ground Control │    │  Local/Remote   │
│  Components     │    │  Agent Manager  │    │  AI Models      │
└─────────────────┘    │  Tool Manager   │    └─────────────────┘
                       │  MCP Service    │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   ChromaDB      │
                       │   Vector Store  │
                       └─────────────────┘
```

## Quick Start

### Prerequisites

1. **Python 3.13+**
2. **Node.js 16+**
3. **Ollama** (for local AI models)
4. **Git**

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd myagent
   ```

2. **Install Python 3.13** (if not already installed)
   - Download from [python.org](https://www.python.org/downloads/)
   - Or use `pyenv` (recommended): `pyenv install 3.13`
   - Verify installation: `python --version` (should show 3.13.x)

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Playwright browsers** (required for Browser Automation tool)
   ```bash
   playwright install
   ```
   This installs the browser binaries needed for the browser automation feature.

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Configuration

Create a `.env` file in the root directory:

```env
# Database settings
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=qwen2.5-coder:7b

# Embedding settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Email settings (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Financial API settings (optional)
ALPHA_VANTAGE_API_KEY=your-api-key
```

### Running the System

1. **Start Ollama** (if not already running)
   ```bash
   ollama serve
   ```

2. **Start the backend**
   ```bash
   python main.py
   ```

3. **Start the frontend** (in a new terminal)
   ```bash
   cd frontend
   npm start
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

## Usage

### 1. RAG Data Management

#### Adding Data
1. Navigate to the RAG Manager
2. Click "Add Data"
3. Choose your data format (JSON, CSV, or Text)
4. Enter your data content
5. Add tags and metadata
6. Submit to create a collection

#### Querying Data
1. Select a collection
2. Click "Query"
3. Enter your search query
4. View results with metadata

### 2. Agent Creation

#### Creating an Agent
1. Go to Agent Manager
2. Click "Create Agent"
3. Configure:
   - Name and description
   - Agent type (RAG, Tool, Hybrid)
   - Model selection
   - Temperature and token limits
   - RAG collections to use
   - Tools to enable
   - System prompt
4. Save the agent

#### Running an Agent
1. Select an agent
2. Click the run button
3. Enter your query
4. View the response

### 3. Tool Configuration

#### Managing Tools
1. Navigate to Tool Manager
2. View all available tools
3. Click "Config" to modify settings
4. Enable/disable tools as needed

#### Browser Automation Tool
The Browser Automation tool uses AI (LangChain) and Playwright to control a web browser and follow natural language instructions.

**Prerequisites:**
- Playwright must be installed: `playwright install`
- The tool requires an LLM provider to be configured (Gemini, Qwen, or Mistral)

**Usage Examples:**
- "Go to google.com and search for 'Python programming'"
- "Navigate to example.com, fill the contact form with name 'John Doe' and email 'john@example.com', then submit"
- "Open github.com, search for 'langchain', and get the first 5 repository names"
- "Go to news.ycombinator.com, scroll down, and take a screenshot"
- "Navigate to amazon.com, search for 'laptop', and get the price of the first result"

**Capabilities:**
- Navigate to websites
- Click buttons and links (by CSS selector or text content)
- Fill forms and input fields
- Extract text and data from pages
- Take screenshots
- Scroll pages (up, down, top, bottom, or by pixels)
- Wait for elements to load
- Select dropdown options
- Get page content (full text or summary)

**How to Use:**
1. Create or edit an agent in Agent Manager
2. Enable the "Browser Automation" tool in the agent's tools list
3. Run the agent with natural language instructions for browser tasks
4. The AI agent will break down your instructions into browser actions and execute them step by step

**Note:** The browser runs in headless mode by default. Screenshots are saved to the current directory with timestamps.

**Using your local browser (Browser Bridge):** When the AI runs in the cloud (e.g. Qwen/Gemini/Mistral APIs), it cannot directly control a browser on the backend server. To have the AI control *your* local Chrome on your machine:

1. On your computer, install: `pip install playwright websockets` and `playwright install chromium`
2. Run the bridge: `python browser_bridge.py` (from the project root). It listens on `ws://0.0.0.0:8765`.
3. In the app (or API), set **browser_bridge_url** to `ws://localhost:8765` (or `ws://YOUR_PC_IP:8765` if the backend is on another machine).
4. Run browser automation as usual. The cloud AI will send commands to the bridge; Playwright on your machine will drive your visible local browser.

### 4. System Monitoring

#### Status Dashboard
1. View system health on the dashboard
2. Monitor Ollama connection
3. Check available models
4. Track collections and agents

## API Endpoints

### System
- `GET /status` - Get system status
- `GET /models` - List available models

### RAG
- `GET /rag/collections` - List collections
- `POST /rag/collections/{name}/data` - Add data
- `POST /rag/collections/{name}/query` - Query collection
- `DELETE /rag/collections/{name}` - Delete collection
- `POST /rag/validate` - Validate data

### Agents
- `GET /agents` - List agents
- `POST /agents` - Create agent
- `GET /agents/{id}` - Get agent details
- `PUT /agents/{id}` - Update agent
- `DELETE /agents/{id}` - Delete agent
- `POST /agents/{id}/run` - Run agent

### Tools
- `GET /tools` - List tools
- `PUT /tools/{id}` - Update tool config

### MCP
- `POST /mcp/start` - Start MCP server

## MCP Protocol

The system implements the Model Context Protocol for stable communication with AI models:

### Message Types
- `initialize` - Client initialization
- `tools/list` - List available tools
- `tools/call` - Execute tool
- `rag/query` - Query RAG collection
- `agent/run` - Run agent
- `ping/pong` - Connection health check

### Connection
- Default port: 8196
- Protocol: WebSocket
- Message format: JSON with length prefix

## Development

### Backend Development

The backend is built with FastAPI and includes:

- **Ground Control**: ChromaDB integration with sentence transformers
- **Agent Manager**: LangChain-based agent orchestration
- **Tool Manager**: Extensible tool system
- **MCP Service**: WebSocket-based protocol implementation

### Frontend Development

The frontend uses React with Material-UI:

- **Dashboard**: System overview and statistics
- **RAG Manager**: Data collection and querying interface
- **Agent Manager**: Agent creation and management
- **Tool Manager**: Tool configuration interface
- **System Status**: Real-time monitoring

### Adding Custom Tools

1. Extend the `ToolManager` class
2. Implement your tool function
3. Register the tool with appropriate metadata
4. The tool will be available in the UI

### Browser Automation Architecture

The Browser Automation tool combines:
- **Playwright**: For browser control and automation
- **LangChain ReAct Agent**: For interpreting natural language instructions
- **Tool System**: Browser actions exposed as LangChain tools

The agent receives natural language instructions, breaks them down into steps, and uses Playwright tools to execute browser actions. The system supports:
- Navigation and page interaction
- Form filling and submission
- Data extraction
- Screenshot capture
- Dynamic element waiting and selection

### Adding New Data Formats

1. Extend the `DataFormat` enum
2. Update the `RAGSystem.validate_data()` method
3. Update the `RAGSystem.process_data()` method
4. Add UI support in the frontend

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Ensure Ollama is running: `ollama serve`
   - Check the base URL in configuration
   - Verify model availability: `ollama list`

2. **ChromaDB Issues**
   - Check disk space
   - Verify permissions for the data directory
   - Restart the application

3. **Frontend Not Loading**
   - Check if the backend is running
   - Verify CORS settings
   - Check browser console for errors

4. **Agent Creation Fails**
   - Verify model name exists in Ollama
   - Check agent configuration
   - Review logs for detailed error messages

5. **Browser Automation Not Working**
   - Ensure Playwright browsers are installed: `playwright install`
   - Verify an LLM provider is configured (Gemini, Qwen, or Mistral)
   - Check that the Browser Automation tool is enabled in your agent
   - Review browser automation logs for specific errors
   - Make sure you have sufficient disk space for screenshots

### Logs

- Backend logs are available in the console
- Frontend errors appear in browser console
- ChromaDB logs are in the data directory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

## Roadmap

- [ ] Multi-modal support (images, audio)
- [ ] Advanced agent workflows
- [ ] Plugin system for tools
- [ ] Distributed deployment
- [ ] Advanced analytics and monitoring
- [ ] Mobile app
- [ ] API rate limiting and authentication
- [ ] Advanced caching strategies 