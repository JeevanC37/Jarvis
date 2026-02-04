"""
Jarvis AI Assistant - Streamlit Chatbot UI

A conversational interface for interacting with the Jarvis AI Assistant.
"""
import streamlit as st
import requests
from typing import List, Dict

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Jarvis AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat UI
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #1e3a5f;
    }
    .chat-message.assistant {
        background-color: #1a1a2e;
    }
    .chat-message .message-content {
        padding: 0.5rem;
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .status-healthy {
        color: #00ff00;
    }
    .status-unhealthy {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict:
    """Check the health of the backend API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def send_message(message: str, history: List[Dict], use_knowledge: bool) -> Dict:
    """Send a message to the Jarvis API."""
    try:
        # Format conversation history
        formatted_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ]
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "conversation_history": formatted_history,
                "use_knowledge_base": use_knowledge
            },
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"response": f"Error: {str(e)}", "sources": None}


def add_knowledge(doc_id: str, content: str) -> Dict:
    """Add knowledge to the database."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/knowledge/add",
            json={
                "doc_id": doc_id,
                "content": content
            },
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_knowledge(query: str) -> Dict:
    """Search the knowledge base."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/knowledge/search",
            json={"query": query, "top_k": 5},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "results": [], "error": str(e)}


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_knowledge" not in st.session_state:
    st.session_state.use_knowledge = True


# Sidebar
with st.sidebar:
    st.markdown("## 🤖 Jarvis AI")
    st.markdown("---")
    
    # Health check
    st.markdown("### System Status")
    health = check_api_health()
    
    if health.get("status") == "healthy":
        st.success("✅ System Healthy")
    elif health.get("status") == "degraded":
        st.warning("⚠️ System Degraded")
    else:
        st.error("❌ System Offline")
        st.info("Make sure the backend is running:\n```\npython main.py\n```")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### Settings")
    st.session_state.use_knowledge = st.toggle(
        "Use Knowledge Base",
        value=st.session_state.use_knowledge,
        help="Enable RAG to use stored knowledge in responses"
    )
    
    st.markdown("---")
    
    # Knowledge Management
    st.markdown("### Knowledge Management")
    
    with st.expander("➕ Add Knowledge"):
        doc_id = st.text_input("Document ID", placeholder="e.g., company_policy")
        content = st.text_area("Content", placeholder="Enter knowledge content...", height=150)
        
        if st.button("Add to Knowledge Base"):
            if doc_id and content:
                result = add_knowledge(doc_id, content)
                if result.get("status") == "success":
                    st.success(f"Added: {doc_id}")
                else:
                    st.error(f"Failed: {result.get('message', 'Unknown error')}")
            else:
                st.warning("Please provide both ID and content")
    
    with st.expander("🔍 Search Knowledge"):
        search_query = st.text_input("Search Query", placeholder="Search knowledge base...")
        
        if st.button("Search"):
            if search_query:
                results = search_knowledge(search_query)
                if results.get("results"):
                    for r in results["results"]:
                        st.markdown(f"**Score: {r['score']:.3f}**")
                        st.markdown(f"> {r['text'][:200]}...")
                        st.markdown("---")
                else:
                    st.info("No results found")
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()


# Main chat interface
st.title("🤖 Jarvis AI Assistant")
st.markdown("*Your personal AI assistant powered by self-hosted LLM with knowledge retrieval*")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source['id']}` (score: {source['score']:.3f})")

# Chat input
if prompt := st.chat_input("Ask Jarvis anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message(
                prompt,
                st.session_state.messages[:-1],  # Exclude current message
                st.session_state.use_knowledge
            )
            
            assistant_message = response.get("response", "I apologize, but I couldn't generate a response.")
            sources = response.get("sources")
            
            st.markdown(assistant_message)
            
            # Show sources if available
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.markdown(f"- `{source['id']}` (score: {source['score']:.3f})")
    
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_message,
        "sources": sources
    })


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Powered by 🦙 LLaMA + 🌲 Pinecone | Built with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
