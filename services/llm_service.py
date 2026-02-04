"""LLM Service for interacting with self-hosted LLaMA via Ollama."""
import httpx
from typing import List, Optional, Generator
import json

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


class LLMService:
    """Service for interacting with self-hosted LLM (LLaMA via Ollama)."""
    
    def __init__(self):
        """Initialize LLM service with Ollama configuration."""
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.client = httpx.Client(timeout=120.0)
    
    def _build_prompt_with_context(
        self, 
        query: str, 
        context_docs: List[dict],
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        """Build a prompt with retrieved context for RAG."""
        # Build context section
        context_text = ""
        if context_docs:
            context_text = "Here is relevant information from the knowledge base:\n\n"
            for i, doc in enumerate(context_docs, 1):
                context_text += f"[{i}] {doc.get('text', '')}\n\n"
        
        # Build conversation history
        history_text = ""
        if conversation_history:
            history_text = "Previous conversation:\n"
            for msg in conversation_history[-5:]:  # Keep last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"{role.capitalize()}: {content}\n"
            history_text += "\n"
        
        # Build final prompt
        system_prompt = """You are Jarvis, a helpful AI assistant for enterprise tasks. 
You provide accurate, helpful, and contextually relevant responses.
If you use information from the knowledge base, reference it naturally in your response.
If you don't know something, say so honestly."""
        
        prompt = f"""{system_prompt}

{context_text}{history_text}User Query: {query}

Please provide a helpful response:"""
        
        return prompt
    
    def generate_response(
        self,
        query: str,
        context_docs: Optional[List[dict]] = None,
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        """Generate a response using the LLM with optional RAG context."""
        prompt = self._build_prompt_with_context(
            query=query,
            context_docs=context_docs or [],
            conversation_history=conversation_history
        )
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "I apologize, but I couldn't generate a response.")
        
        except httpx.HTTPError as e:
            return f"Error communicating with LLM service: {str(e)}"
    
    def generate_response_stream(
        self,
        query: str,
        context_docs: Optional[List[dict]] = None,
        conversation_history: Optional[List[dict]] = None
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM."""
        prompt = self._build_prompt_with_context(
            query=query,
            context_docs=context_docs or [],
            conversation_history=conversation_history
        )
        
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True
                }
            ) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
        
        except httpx.HTTPError as e:
            yield f"Error: {str(e)}"
    
    def check_health(self) -> dict:
        """Check if the Ollama service is running."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return {
                "status": "healthy",
                "available_models": model_names,
                "configured_model": self.model
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
