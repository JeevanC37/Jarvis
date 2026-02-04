
# 🤖 Jarvis AI Assistant

A personal AI assistant powered by a **self-hosted LLM (LLaMA)** with **Pinecone vector database** for knowledge retrieval (RAG) and a **Streamlit chatbot UI**.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Jarvis AI Assistant                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   Streamlit  │────▶│   FastAPI    │────▶│   Ollama     │   │
│   │   Chat UI    │     │   Backend    │     │   (LLaMA)    │   │
│   └──────────────┘     └──────┬───────┘     └──────────────┘   │
│                               │                                  │
│                               ▼                                  │
│                        ┌──────────────┐                         │
│                        │   Pinecone   │                         │
│                        │ Vector Store │                         │
│                        └──────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Features

- **Self-hosted LLM**: Uses Ollama with LLaMA model for privacy-first AI
- **RAG (Retrieval Augmented Generation)**: Contextual responses using your knowledge base
- **Vector Database**: Pinecone for efficient semantic search
- **Conversational Memory**: Maintains chat context across messages
- **Knowledge Management**: Add, search, and delete documents from UI
- **Streaming Responses**: Real-time response generation
- **REST API**: Full API for programmatic access

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Ollama** - Self-hosted LLM runtime
3. **Pinecone Account** - Free tier available

### 1. Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Start Ollama service
ollama serve

# Pull LLaMA model
ollama pull llama2
```

### 2. Set Up Pinecone

1. Create account at [pinecone.io](https://www.pinecone.io/)
2. Create a new project
3. Get your API key from the dashboard

### 3. Configure Environment

```bash
# Clone and enter directory
cd jarvis

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Update `.env` with your values:
```env
PINECONE_API_KEY=your-actual-api-key
PINECONE_INDEX_NAME=jarvis-knowledge
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### 4. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 5. Run the Application

```bash
# Terminal 1: Start the API backend
python main.py

# Terminal 2: Start the Streamlit UI
streamlit run app.py
```

The application will be available at:
- **Chat UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 📚 API Reference

### Chat Endpoints

#### POST `/chat`
Send a message and get a response.

```json
{
  "message": "What is our company policy on remote work?",
  "conversation_history": [],
  "use_knowledge_base": true
}
```

#### POST `/chat/stream`
Stream a response in real-time.

### Knowledge Management

#### POST `/knowledge/add`
Add a document to the knowledge base.

```json
{
  "doc_id": "remote_policy",
  "content": "Our company allows remote work up to 3 days per week...",
  "metadata": {"category": "HR", "version": "2024"}
}
```

#### POST `/knowledge/search`
Search the knowledge base.

```json
{
  "query": "remote work policy",
  "top_k": 5
}
```

#### DELETE `/knowledge/{doc_id}`
Delete a document from the knowledge base.

## 📁 Project Structure

```
jarvis/
├── main.py              # FastAPI backend server
├── app.py               # Streamlit chat UI
├── config.py            # Configuration management
├── models.py            # Pydantic models
├── ingest.py            # Knowledge ingestion utilities
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── .env                 # Your configuration (create this)
└── services/
    ├── __init__.py
    ├── llm_service.py   # Ollama/LLaMA integration
    └── vector_db.py     # Pinecone integration
```

## 🔧 Ingesting Knowledge

### From Python

```python
from ingest import ingest_text, ingest_file, ingest_directory

# Add text directly
ingest_text(
    text="Company vacation policy allows 20 days PTO per year...",
    doc_id="vacation_policy",
    metadata={"category": "HR"}
)

# Ingest a file
ingest_file("documents/handbook.txt")

# Ingest entire directory
ingest_directory("documents/", extensions=[".txt", ".md"], recursive=True)
```

### From Command Line

```bash
# Ingest a single file
python ingest.py documents/policy.txt

# Ingest a directory recursively
python ingest.py documents/ -r -e .txt .md
```

## 🛠️ Customization

### Using Different LLM Models

```bash
# Pull a different model
ollama pull mistral
ollama pull codellama

# Update .env
OLLAMA_MODEL=mistral
```

### Adjusting Embedding Model

Edit `config.py`:
```python
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Larger, more accurate
EMBEDDING_DIMENSION = 768
```

> **Note**: If you change embedding dimensions, you'll need to recreate your Pinecone index.

## 🐛 Troubleshooting

### Ollama Connection Error
```
Error communicating with LLM service
```
- Ensure Ollama is running: `ollama serve`
- Check if model is downloaded: `ollama list`

### Pinecone Authentication Error
- Verify your API key in `.env`
- Check if you're using the correct environment

### Slow Responses
- Consider using a smaller model: `ollama pull llama2:7b`
- Reduce `top_k` for knowledge retrieval

## 📄 License

MIT License - Feel free to use and modify for your enterprise needs!

---

Built with ❤️ for the "Code Meets Co-Pilot" workshop
=======
# Jarvis
>>>>>>> f1d3b10a5d17543d7338d7282022e6b9cd079b7c
