"""Knowledge ingestion utilities for bulk loading documents into Jarvis."""
import os
import uuid
from typing import List, Optional
from pathlib import Path

from services.vector_db import get_vector_db_service


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better embedding quality.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundary
            for sep in ['. ', '! ', '? ', '\n\n', '\n', ' ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks


def ingest_text(
    text: str,
    doc_id: str,
    metadata: Optional[dict] = None,
    chunk: bool = True
) -> dict:
    """
    Ingest a text document into the knowledge base.
    
    Args:
        text: The text content to ingest
        doc_id: Unique identifier for the document
        metadata: Additional metadata to store
        chunk: Whether to split text into chunks
    
    Returns:
        Ingestion result with status and chunk count
    """
    vector_db = get_vector_db_service()
    
    if chunk:
        chunks = chunk_text(text)
    else:
        chunks = [text]
    
    ingested_count = 0
    for i, chunk_text_content in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}" if len(chunks) > 1 else doc_id
        chunk_metadata = {
            **(metadata or {}),
            "parent_doc_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        
        vector_db.store_knowledge(
            doc_id=chunk_id,
            text=chunk_text_content,
            metadata=chunk_metadata
        )
        ingested_count += 1
    
    return {
        "status": "success",
        "doc_id": doc_id,
        "chunks_created": ingested_count
    }


def ingest_file(file_path: str, metadata: Optional[dict] = None) -> dict:
    """
    Ingest a text file into the knowledge base.
    
    Args:
        file_path: Path to the file to ingest
        metadata: Additional metadata to store
    
    Returns:
        Ingestion result
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read file content
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create document ID from filename
    doc_id = path.stem.replace(' ', '_').lower()
    
    # Add file metadata
    file_metadata = {
        **(metadata or {}),
        "source_file": str(path.name),
        "file_type": path.suffix
    }
    
    return ingest_text(content, doc_id, file_metadata)


def ingest_directory(
    directory_path: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> dict:
    """
    Ingest all text files from a directory.
    
    Args:
        directory_path: Path to the directory
        extensions: List of file extensions to include (e.g., ['.txt', '.md'])
        recursive: Whether to search subdirectories
    
    Returns:
        Summary of ingestion results
    """
    if extensions is None:
        extensions = ['.txt', '.md', '.rst', '.json']
    
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    results = {
        "total_files": 0,
        "successful": 0,
        "failed": 0,
        "errors": []
    }
    
    pattern = "**/*" if recursive else "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            results["total_files"] += 1
            try:
                ingest_file(str(file_path), {"source_directory": str(directory)})
                results["successful"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "file": str(file_path),
                    "error": str(e)
                })
    
    return results


# CLI for ingestion
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into Jarvis knowledge base")
    parser.add_argument("path", help="File or directory path to ingest")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively ingest directories")
    parser.add_argument("--extensions", "-e", nargs="+", default=[".txt", ".md"], help="File extensions to include")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        result = ingest_file(str(path))
        print(f"Ingested file: {result}")
    elif path.is_dir():
        result = ingest_directory(str(path), args.extensions, args.recursive)
        print(f"Directory ingestion complete: {result}")
    else:
        print(f"Path not found: {args.path}")
