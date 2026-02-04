"""Vector database service using Pinecone for knowledge storage and retrieval."""
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION
)


class VectorDBService:
    """Service for managing vector embeddings in Pinecone."""
    
    def __init__(self):
        """Initialize Pinecone client and embedding model."""
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.index_name = PINECONE_INDEX_NAME
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Created new Pinecone index: {self.index_name}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def store_knowledge(self, doc_id: str, text: str, metadata: Optional[dict] = None):
        """Store a document with its embedding in Pinecone."""
        embedding = self.get_embedding(text)
        
        vector_data = {
            "id": doc_id,
            "values": embedding,
            "metadata": {
                "text": text,
                **(metadata or {})
            }
        }
        
        self.index.upsert(vectors=[vector_data])
        return {"status": "success", "doc_id": doc_id}
    
    def search_knowledge(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for relevant documents based on query."""
        query_embedding = self.get_embedding(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        retrieved_docs = []
        for match in results.matches:
            retrieved_docs.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata
            })
        
        return retrieved_docs
    
    def delete_knowledge(self, doc_id: str):
        """Delete a document from the vector store."""
        self.index.delete(ids=[doc_id])
        return {"status": "deleted", "doc_id": doc_id}


# Singleton instance
_vector_db_service: Optional[VectorDBService] = None


def get_vector_db_service() -> VectorDBService:
    """Get or create the VectorDB service instance."""
    global _vector_db_service
    if _vector_db_service is None:
        _vector_db_service = VectorDBService()
    return _vector_db_service
