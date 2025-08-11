import streamlit as st
import os
import tempfile
from processing.chunkings import PDFChunker
from processing.embeddings import EmbeddingManager
from processing.vectorstore import FAISSVectorStore
from processing.retrievals import RetrievalManager
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration with dark theme
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="assets/mpg_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
        color: #fafafa;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4a4a4a;
    }
    .stFileUploader > div > div {
        background-color: #262730;
        border: 1px solid #4a4a4a;
    }
    .stExpander > div > div {
        background-color: #262730;
        border: 1px solid #4a4a4a;
    }
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4a4a4a;
    }
    h1, h2, h3 {
        color: #fafafa;
    }
    .chat-message {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .user-message {
        border-left-color: #2196F3;
    }
    .assistant-message {
        border-left-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retrieval_manager' not in st.session_state:
    st.session_state.retrieval_manager = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

def initialize_system():
    """Initialize the RAG system components using environment variables."""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        st.error("🚨 **OpenAI API Key Missing!**")
        st.error("Please add your OpenAI API key to the `.env` file:")
        st.code("OPENAI_API_KEY=your_api_key_here")
        st.error("Without a valid API key, the system cannot process PDFs or answer questions.")
        return False
    
    try:
        # Initialize vector store
        vector_store = FAISSVectorStore()
        
        # Try to load existing data only if directory exists and has content
        if os.path.exists("vector_store") and os.listdir("vector_store"):
            try:
                vector_store.load("vector_store")
                stats = vector_store.get_stats()
                if stats['total_items'] > 0:
                    st.success("✅ Loaded existing knowledge base")
                else:
                    st.info("📝 No content in existing knowledge base. Upload PDFs to get started.")
            except Exception as e:
                st.warning("⚠️ Could not load existing knowledge base. Starting fresh.")
                vector_store = FAISSVectorStore()  # Fresh instance
        else:
            st.info("📝 No existing knowledge base found. Upload PDFs to get started.")
        
        # Initialize retrieval manager
        retrieval_manager = RetrievalManager(openai_api_key, vector_store)
        
        # Store in session state
        st.session_state.vector_store = vector_store
        st.session_state.retrieval_manager = retrieval_manager
        st.session_state.system_initialized = True
        
        return True
    except Exception as e:
        st.error(f"🚨 Error initializing system: {e}")
        return False

def process_pdf(uploaded_file):
    """Process uploaded PDF and add to vector store."""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        # Initialize components
        chunker = PDFChunker()
        embedding_manager = EmbeddingManager(openai_api_key)
        
        # Process PDF
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Extracting content from PDF...")
        progress_bar.progress(25)
        
        # Extract chunks
        chunks = chunker.process_pdf(pdf_path)
        
        status_text.text("🔄 Generating embeddings...")
        progress_bar.progress(50)
        
        # Get embeddings
        processed_chunks = embedding_manager.process_chunks(chunks)
        
        status_text.text("🔄 Adding to knowledge base...")
        progress_bar.progress(75)
        
        # Add to vector store
        if st.session_state.vector_store:
            st.session_state.vector_store.add_text_chunks(processed_chunks['text_chunks'])
            st.session_state.vector_store.add_images(processed_chunks['images'])
            st.session_state.vector_store.add_tables(processed_chunks['tables'])
            
            # Save vector store
            st.session_state.vector_store.save("vector_store")
            
            # Add to uploaded files list
            st.session_state.uploaded_files.append(uploaded_file.name)
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            st.success(f"✅ Successfully processed {uploaded_file.name}")
            return True
        
        # Clean up temporary file
        os.unlink(pdf_path)
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return False

def add_to_chat_history(question, answer):
    """Add question and answer to chat history."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_entry = {
        'timestamp': timestamp,
        'question': question,
        'answer': answer
    }
    st.session_state.chat_history.append(chat_entry)
    
    # Add to vector store for persistent memory
    if st.session_state.vector_store and st.session_state.retrieval_manager:
        try:
            # Create chat chunk for vector store
            chat_content = f"Q: {question}\nA: {answer}"
            chat_embedding = st.session_state.retrieval_manager.embedding_manager.get_embeddings([chat_content])[0]
            
            chat_chunk = {
                'type': 'chat',
                'content': chat_content,
                'question': question,
                'answer': answer,
                'timestamp': timestamp,
                'embedding': chat_embedding,
                'metadata': {
                    'chunk_type': 'chat',
                    'conversation_id': len(st.session_state.chat_history),
                    'timestamp': timestamp
                }
            }
            
            # Add to vector store
            st.session_state.vector_store.add_chat_chunks([chat_chunk])
            
            # Save vector store to persist chat history
            st.session_state.vector_store.save("vector_store")
            
        except Exception as e:
            st.error(f"Error adding chat to vector store: {e}")

def display_chat_message(question, answer, is_user=True):
    """Display a chat message with proper styling."""
    message_class = "user-message" if is_user else "assistant-message"
    icon = "👤" if is_user else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{icon} {'You' if is_user else 'Assistant'}:</strong><br>
        {question if is_user else answer}
    </div>
    """, unsafe_allow_html=True)

def display_citation_info(citation_text):
    """Display citation information in a user-friendly way."""
    if citation_text:
        st.info(f"📚 **Sources:** {citation_text}")
        st.markdown("💡 *Click '🔍 View Search Results & Citations' above to see detailed sources*")

def main():
    # Initialize system on first load
    if not st.session_state.system_initialized:
        if not initialize_system():
            # If initialization failed (e.g., no API key), show error and stop
            st.title("📚 PDF RAG System")
            st.markdown("Upload your PDFs and ask questions with AI-powered retrieval and citations.")
            st.error("🚨 **System cannot start without a valid OpenAI API key.**")
            st.error("Please fix the API key issue and refresh the page.")
            return
    
    # Main layout
    st.title("📚 PDF RAG System")
    st.markdown("Upload your PDFs and ask questions with AI-powered retrieval and citations.")
    
    # Sidebar for file upload and chat history
    with st.sidebar:
        # Add logo at the top of sidebar
        st.image("assets/image.png", width=200)
        st.markdown("---")
        
        st.header("📁 Upload PDFs")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF to process and add to the knowledge base"
        )
        
        if uploaded_file:
            # Check if file already exists and knowledge base has content
            if uploaded_file.name in st.session_state.uploaded_files and st.session_state.vector_store:
                stats = st.session_state.vector_store.get_stats()
                if stats['total_items'] > 0:
                    st.warning(f"⚠️ {uploaded_file.name} is already in the knowledge base!")
                    st.info("Uploading again will add duplicate content. Use 'Clear All Data' to start fresh.")
            
            if st.button("🔄 Process PDF"):
                if process_pdf(uploaded_file):
                    st.rerun()
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📋 Uploaded Files")
            for i, file_name in enumerate(st.session_state.uploaded_files):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {file_name}")
                with col2:
                    if st.button(f"🗑️", key=f"remove_file_{i}"):
                        # Remove file from list (note: this doesn't remove from vector store yet)
                        st.session_state.uploaded_files.pop(i)
                        st.rerun()
        
        # Vector store stats and management
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_stats()
            st.subheader("📊 Knowledge Base Stats")
            st.write(f"Text chunks: {stats['text_chunks']}")
            st.write(f"Images: {stats['images']}")
            st.write(f"Tables: {stats['tables']}")
            st.write(f"Chat chunks: {stats['chat_chunks']}")
            st.write(f"Total items: {stats['total_items']}")
            
            # Show last update time
            if st.session_state.uploaded_files:
                st.write(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"📚 Documents: {len(st.session_state.uploaded_files)}")
            
            # Vector store management
            st.subheader("🗄️ Knowledge Base Management")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Clear All Data", key="clear_vector_store"):
                    if st.session_state.vector_store:
                        # Clear vector store
                        st.session_state.vector_store.clear_all()
                        st.session_state.uploaded_files = []
                        
                        # Remove vector store directory from disk
                        import shutil
                        if os.path.exists("vector_store"):
                            shutil.rmtree("vector_store")
                        
                        st.success("✅ Knowledge base completely cleared!")
                        st.rerun()
            with col2:
                if st.button("🔄 Reset System", key="reset_system"):
                    st.session_state.system_initialized = False
                    st.session_state.chat_history = []
                    st.session_state.uploaded_files = []
                    st.rerun()
            with col3:
                if st.button("🔧 System Check", key="system_check"):
                    st.session_state.show_diagnostics = True
                    st.rerun()
        
        # System diagnostics
        if st.session_state.get('show_diagnostics', False):
            with st.expander("🔧 System Diagnostics", expanded=True):
                st.write("**Checking system components...**")
                
                # Check fitz table extraction
                try:
                    import fitz
                    # Test table extraction capability
                    test_doc = fitz.open()
                    test_page = test_doc.new_page()
                    has_table_finder = hasattr(test_page, 'find_tables')
                    test_doc.close()
                    
                    if has_table_finder:
                        st.success("✅ PyMuPDF Table Extraction: Available")
                    else:
                        st.warning("⚠️ PyMuPDF Table Extraction: Limited (will use pattern matching)")
                except ImportError:
                    st.error("❌ PyMuPDF: Not installed")
                
                # Check other dependencies
                try:
                    import fitz
                    st.success("✅ PyMuPDF: Available")
                except ImportError:
                    st.error("❌ PyMuPDF: Not installed")
                
                try:
                    import openai
                    st.success("✅ OpenAI: Available")
                except ImportError:
                    st.error("❌ OpenAI: Not installed")
                
                if st.button("Close Diagnostics"):
                    st.session_state.show_diagnostics = False
                    st.rerun()
        
        # Chat history
        st.header(" Recent Chats")
        
        # Chat history management
        if st.session_state.chat_history:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All", key="clear_all_history"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                # Export chat history
                chat_text = ""
                for entry in st.session_state.chat_history:
                    chat_text += f"Q: {entry['question']}\nA: {entry['answer']}\n\n"
                
                st.download_button(
                    label="📥 Export",
                    data=chat_text,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="export_history"
                )
        
        if st.session_state.chat_history:
            for i, entry in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
                with st.expander(f"{entry['timestamp']} - {entry['question'][:30]}..."):
                    st.write(f"**Q:** {entry['question']}")
                    st.write(f"**A:** {entry['answer'][:100]}...")
                    
                    # Delete individual entry button
                    if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                        st.session_state.chat_history.pop(-(i+1))
                        st.rerun()
        else:
            st.write("No chat history yet.")
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Check if system is initialized
    if not st.session_state.vector_store or not st.session_state.retrieval_manager:
        st.warning("⚠️ System not initialized. Please check your .env file for OPENAI_API_KEY.")
        return
    
    # Check if there's data in the vector store
    stats = st.session_state.vector_store.get_stats()
    if stats['total_items'] == 0:
        st.info("📚 No documents in the knowledge base. Please upload a PDF first.")
        return
    
    # Display recent chat history in main area
    if st.session_state.chat_history:
        st.subheader(" Recent Conversation")
        
        # Display citation info for the latest response
        if 'last_search_results' in st.session_state:
            display_citation_info(st.session_state.last_search_results['citations'])
        
        # Display persistent search results if available
        if 'last_search_results' in st.session_state:
            col1, col2 = st.columns([3, 1])
            with col1:
                with st.expander("🔍 View Search Results & Citations", expanded=False):
                    st.write(f"**Question:** {st.session_state.last_search_results['question']}")
                    st.write(f"**Citations:** {st.session_state.last_search_results['citations']}")
                    st.divider()
                    
                    for i, result in enumerate(st.session_state.last_search_results['results'], 1):
                        st.write(f"**Result {i}** (Score: {result.get('combined_score', 0):.3f})")
                        if result['type'] == 'text':
                            st.write(f"📄 Page {result['page']}, Lines {result['line_start']}-{result['line_end']}")
                            st.text_area(f"Content {i}", result['content'], height=100, key=f"content_{i}")
                        elif result['type'] == 'image':
                            st.write(f"🖼️ Page {result['page']}, Image {result['image_index']}")
                            st.write(f"Description: {result['description']}")
                        elif result['type'] == 'table':
                            st.write(f"📊 Page {result['page']}, Table {result['table_index']}")
                            st.text_area(f"Table {i}", result['content'], height=100, key=f"table_{i}")
                        st.divider()
            with col2:
                if st.button("🗑️ Clear Results", key="clear_search_results"):
                    del st.session_state.last_search_results
                    st.rerun()
        
        # Add conversation summary
        if len(st.session_state.chat_history) > 2:
            with st.expander(" Conversation Summary"):
                topics = []
                for entry in st.session_state.chat_history[-3:]:
                    topics.append(f"• {entry['question'][:50]}...")
                st.write("**Recent topics discussed:**")
                for topic in topics:
                    st.write(topic)
        
        # Show last 5 entries with better formatting
        for i, entry in enumerate(st.session_state.chat_history[-5:]):
            # User question
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {entry['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant answer
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong> Assistant:</strong><br>
                {entry['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Quick follow-up suggestions
            if i == len(st.session_state.chat_history[-5:]) - 1:  # Only for the latest entry
                st.markdown("**💡 Quick follow-ups:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔍 Tell me more", key=f"followup_more_{i}"):
                        st.session_state.quick_question = f"Tell me more about {entry['question']}"
                        st.rerun()
                with col2:
                    if st.button("📊 Show examples", key=f"followup_examples_{i}"):
                        st.session_state.quick_question = f"Can you provide examples related to {entry['question']}"
                        st.rerun()
                with col3:
                    if st.button("❓ What else?", key=f"followup_else_{i}"):
                        st.session_state.quick_question = f"What else should I know about this topic?"
                        st.rerun()
            
            st.divider()
    
    # Chat input at bottom
    st.markdown("---")
    
    # Show indicator if quick question is set (outside form)
    default_question = st.session_state.get('quick_question', "")
    if default_question:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"💡 Quick question set: {default_question}")
        with col2:
            if st.button("🗑️ Clear", key="clear_quick_question"):
                del st.session_state.quick_question
                st.rerun()
    
    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a question about your documents:",
            value=default_question,
            placeholder="e.g., What are the main topics discussed in the document?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.form_submit_button("🔍 Ask", use_container_width=True)
        
        # Clear quick question after form is submitted
        if ask_button and 'quick_question' in st.session_state:
            del st.session_state.quick_question
    
    # Process question
    if user_question and ask_button:
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                # Get context and citations
                context, results = st.session_state.retrieval_manager.get_context_for_qa(
                    user_question, 
                    chat_history=st.session_state.chat_history
                )
                citations = st.session_state.retrieval_manager.format_citations(results)
                
                # Generate answer with chat history
                answer = st.session_state.retrieval_manager.answer_question(
                    user_question, 
                    context, 
                    citations, 
                    chat_history=st.session_state.chat_history
                )
                
                # Add to chat history with search results
                add_to_chat_history(user_question, answer)
                
                # Store search results in session state for persistent display
                st.session_state.last_search_results = {
                    'question': user_question,
                    'citations': citations,
                    'results': results
                }
                
                # Display the new conversation
                st.subheader("💬 Latest Response")
                display_chat_message(user_question, "", is_user=True)
                display_chat_message("", answer, is_user=False)
                
                # Form will clear automatically due to clear_on_submit=True
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing question: {e}")

if __name__ == "__main__":
    main() 