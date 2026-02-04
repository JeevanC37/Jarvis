"""FastAPI Backend for Jarvis AI Assistant."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from config import APP_HOST, APP_PORT
from models import (
    ChatRequest,
    ChatResponse,
    KnowledgeAddRequest,
    KnowledgeSearchRequest,
    KnowledgeResponse,
    HealthResponse
)
from services.llm_service import get_llm_service
from services.vector_db import get_vector_db_service

# Initialize FastAPI app
app = FastAPI(
    title="Jarvis AI Assistant",
    description="Personal AI Assistant powered by self-hosted LLM with RAG capabilities",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with welcome message."""
    return {
        "message": "Welcome to Jarvis AI Assistant",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check health of all services."""
    llm_service = get_llm_service()
    llm_health = llm_service.check_health()
    
    # Check vector DB
    try:
        vector_db = get_vector_db_service()
        vector_db_status = "healthy"
    except Exception as e:
        vector_db_status = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if llm_health["status"] == "healthy" and vector_db_status == "healthy" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        llm_status=llm_health,
        vector_db_status=vector_db_status
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with Jarvis AI Assistant.
    
    Uses RAG (Retrieval Augmented Generation) to provide contextually
    relevant responses based on the knowledge base.
    """
    try:
        llm_service = get_llm_service()
        sources = []
        context_docs = []
        
        # Retrieve relevant context if knowledge base is enabled
        if request.use_knowledge_base:
            try:
                vector_db = get_vector_db_service()
                context_docs = vector_db.search_knowledge(request.message, top_k=3)
                sources = [{"id": doc["id"], "score": doc["score"]} for doc in context_docs]
            except Exception as e:
                print(f"Warning: Could not retrieve from knowledge base: {e}")
        
        # Convert conversation history
        history = None
        if request.conversation_history:
            history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
        
        # Generate response
        response = llm_service.generate_response(
            query=request.message,
            context_docs=context_docs,
            conversation_history=history
        )
        
        return ChatResponse(response=response, sources=sources if sources else None)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream chat response from Jarvis AI Assistant.
    
    Returns a streaming response for real-time text generation.
    """
    try:
        llm_service = get_llm_service()
        context_docs = []
        
        # Retrieve relevant context if knowledge base is enabled
        if request.use_knowledge_base:
            try:
                vector_db = get_vector_db_service()
                context_docs = vector_db.search_knowledge(request.message, top_k=3)
            except Exception:
                pass
        
        # Convert conversation history
        history = None
        if request.conversation_history:
            history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
        
        def generate():
            for chunk in llm_service.generate_response_stream(
                query=request.message,
                context_docs=context_docs,
                conversation_history=history
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/add", response_model=KnowledgeResponse, tags=["Knowledge"])
async def add_knowledge(request: KnowledgeAddRequest):
    """Add a document to the knowledge base."""
    try:
        vector_db = get_vector_db_service()
        result = vector_db.store_knowledge(
            doc_id=request.doc_id,
            text=request.content,
            metadata=request.metadata
        )
        
        return KnowledgeResponse(
            status="success",
            message=f"Document '{request.doc_id}' added to knowledge base",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/search", tags=["Knowledge"])
async def search_knowledge(request: KnowledgeSearchRequest):
    """Search the knowledge base."""
    try:
        vector_db = get_vector_db_service()
        results = vector_db.search_knowledge(request.query, top_k=request.top_k)
        
        return {
            "status": "success",
            "query": request.query,
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/knowledge/{doc_id}", response_model=KnowledgeResponse, tags=["Knowledge"])
async def delete_knowledge(doc_id: str):
    """Delete a document from the knowledge base."""
    try:
        vector_db = get_vector_db_service()
        result = vector_db.delete_knowledge(doc_id)
        
        return KnowledgeResponse(
            status="success",
            message=f"Document '{doc_id}' deleted from knowledge base",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True
    )
