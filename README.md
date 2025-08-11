# PDF RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for PDF documents that supports text, images, and tables with hybrid search capabilities and detailed citations.

## Features

- **Multimodal Processing**: Extracts and processes text, images, and tables from PDFs
- **Hybrid Search**: Combines dense embeddings and keyword search for better recall
- **Detailed Citations**: Provides page numbers and line references for all answers
- **Chat History**: Persistent chat history with the ability to view and manage conversations
- **FAISS Vector Store**: Fast and efficient similarity search using FAISS
- **OpenAI Integration**: Uses OpenAI's Vision model for image understanding and GPT for text generation

### Evaluations: 8 of 10 manual test cases performed successfully.

## Demo Link: 


https://github.com/user-attachments/assets/76946046-27c9-4963-9a52-f6bc754d0309


## Architecture

The system is organized into modular components:

```
rag_withChatHistory_MPG/
├── processing/
│   ├── __init__.py
│   ├── chunkings.py      # PDF chunking and extraction logic
│   ├── embeddings.py     # Embedding generation for all content types
│   ├── vectorstore.py    # FAISS vector store management
│   └── retrievals.py     # Hybrid search and retrieval logic
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag_multimodal_MPG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API Key**:
   - Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Copy `env_template.txt` to `.env` and add your API key
   - Example: `OPENAI_API_KEY=sk-your-actual-key-here`

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Set up environment**:
   - Add your OpenAI API key to the `.env` file

3. **Upload PDFs**:
   - Use the file uploader in the left sidebar
   - Click " Process PDF" to extract and index the content
   - The system will process text, images, and tables automatically

4. **Ask questions**:
   - Type your question in the chat interface at the bottom
   - Click "🔍 Ask" to get AI-powered answers
   - The system will search across all content types with detailed citations
   - Use follow-up suggestions for quick related questions
   - Chat history provides context for better follow-up answers

5. **Manage knowledge base**:
   - View statistics in the sidebar
   - Use "Clear All Data" to remove all documents and start fresh
   - Use "Reset System" to clear chat history and restart
   - Uploaded files are listed with individual remove options

## System Components

### 1. PDF Chunking (`chunkings.py`)
- Extracts text with line-level tracking
- Identifies and extracts images with position data
- Detects and extracts tables using fitz (PyMuPDF) table finder
- Maintains detailed metadata for citations

### 2. Embedding Generation (`embeddings.py`)
- Text embeddings using OpenAI's text-embedding-3-small
- Image descriptions using GPT-o4-mini
- Table embeddings with enhanced context
- Unified embedding approach for all content types

### 3. Vector Store (`vectorstore.py`)
- Separate FAISS indexes for text, images, and tables
- Efficient similarity search with metadata tracking
- Persistent storage and loading capabilities
- Statistics and monitoring features

### 4. Hybrid Retrieval (`retrievals.py`)
- Combines dense embeddings and keyword search
- Configurable weighting between search methods
- Detailed citation formatting
- Context-aware question answering

## Key Features

### Hybrid Search
The system uses a combination of:
- **Dense Search**: Semantic similarity using embeddings
- **Keyword Search**: Traditional term matching
- **Combined Scoring**: Weighted combination for optimal results

### Detailed Citations
Every answer includes:
- Page numbers
- Line ranges for text
- Image indices for visual content
- Table references for tabular data

### Chat History & Follow-ups
- Persistent conversation history in the left sidebar
- Expandable chat entries with timestamps
- Individual message deletion and bulk clear options
- Recent conversations displayed in main area
- **Follow-up suggestions** with quick action buttons
- **Conversation context** used for better follow-up answers
- **Chat history export** functionality
- **Conversation summary** for easy reference

### User Interface
- Dark theme matching modern chat applications
- Clean, minimalist design with intuitive navigation
- Automatic system initialization from environment variables
- Real-time chat interface with message styling

## Configuration

### Chunking Parameters
- `chunk_size`: Maximum size of text chunks (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks (default: 200 characters)

### Search Parameters
- `k`: Number of results to retrieve (default: 5)
- `alpha`: Weight for dense vs keyword search (default: 0.7)

### Model Configuration
- Text embeddings: `text-embedding-3-small`
- Image understanding: `gpt-o4-mini` (updated from deprecated `gpt-4-vision-preview`)
- Text generation: `gpt-o4-mini`

## File Structure

The system creates the following files during operation:
- `vector_store/`: Directory containing FAISS indexes and metadata
- `vector_store/text_index.faiss`: Text embeddings index
- `vector_store/image_index.faiss`: Image embeddings index
- `vector_store/table_index.faiss`: Table embeddings index
- `vector_store/*_metadata.pkl`: Metadata files for each content type

### Data Persistence
- **Vector data is stored persistently** in the `vector_store/` directory
- **New PDFs are added** to existing knowledge base (not replaced)
- **Use "Clear All Data"** to start fresh with new documents
- **Chat history** is maintained during the session

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   - Ensure your API key is valid and has sufficient credits
   - Check that you have access to the required models

2. **PDF Processing Errors**:
   - Ensure PDFs are not password-protected
   - Check that PDFs contain extractable text
   - Some scanned PDFs may not work without OCR

3. **Memory Issues**:
   - Large PDFs may require significant memory
   - Consider processing smaller documents or increasing system memory

4. **FAISS Installation**:
   - On Windows, use `faiss-cpu` instead of `faiss`
   - For GPU support, install `faiss-gpu`

5. **Table Extraction**:
   - Uses **fitz (PyMuPDF) table finder** for native table detection
   - Automatically falls back to pattern matching if table finder fails
   - Consistent with text and image extraction (all using PyMuPDF)

## Performance Considerations

- **Processing Time**: Large PDFs with many images may take several minutes
- **Memory Usage**: Vector store size grows with document content
- **Search Speed**: FAISS provides fast similarity search even with large datasets
- **API Costs**: OpenAI API usage depends on document size and number of queries

## Future Enhancements

- Support for more document formats (DOCX, PPTX, etc.)
- Advanced OCR for scanned documents
- Multi-language support
- Custom embedding models
- Advanced filtering and faceted search
- Export capabilities for search results

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
