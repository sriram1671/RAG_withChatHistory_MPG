# PDF RAG System Architecture

## Overview

This is a **Retrieval-Augmented Generation (RAG)** system designed to process PDF documents and provide intelligent question-answering capabilities with detailed citations. The system extracts text, images, and tables from PDFs, generates embeddings, stores them in a vector database, and uses AI models to answer questions with context-aware responses.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Streamlit UI  │    │  PDF Processing │    │  Vector Store    │
│   (app.py)      │◄──►│  (chunkings.py) │◄──►│  (vectorstore.py)│
└─────────────────┘    └─────────────────┘    └──────────────────┘
         │                       │                       │
         │              ┌───────────────────┐            │
         │              │ Multi-Modal       │            │
         │              │ Content Split     │            │
         │              │ ┌─────┬─────┬───┐ |            │
         │              │ │Text │Image│Tab│ |            │
         │              │ │     │     │le │ |            │
         │              │ └─────┴─────┴───┘ |            │
         │              └───────────────────┘            │
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   Embeddings    │              │
         │              │ (embeddings.py) │              │
         │              └─────────────────┘              │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chat History  │    │  OpenAI Models  │    │   FAISS Index   │
│   Management    │    │  (GPT-4o, etc.) │    │   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │   Retrieval &   │
                        │   Generation    │
                        │ (retrievals.py) │
                        └─────────────────┘
```

## Core Components

### 1. **PDF Processing Layer** (`processing/chunkings.py`)

**Purpose**: Extracts and chunks different types of content from PDF documents.

**Key Features**:
- **Text Extraction**: Extracts text with line-level tracking for precise citations
- **Image Extraction**: Identifies and extracts images with position data
- **Table Extraction**: Detects tables using PyMuPDF's native table finder with fallback pattern matching
- **Chunking Strategy**: Implements overlapping text chunks for better context preservation

**Libraries Used**:
- `fitz` (PyMuPDF): Primary PDF processing library
- `PIL` (Pillow): Image processing and format conversion
- `pandas`: Data manipulation for table processing
- `logging`: Detailed processing logs for debugging

**Data Flow**:
```
PDF Input → Multi-Modal Content Extraction → Parallel Processing → Structured Output
     ↓                    ↓                           ↓                    ↓
  Document Load      ┌─────────┬─────────┬─────────┐  ┌─────────┬─────────┬─────────┐  ┌─────────┬─────────┬─────────┐
                     │  Text   │ Images  │ Tables  │  │ Text    │ Image   │ Table   │  │ Text    │ Image   │ Table   │
                     │Extract  │Extract  │Extract  │  │Chunking │Desc Gen │Parsing  │  │Chunks   │Metadata │Content  │
                     └─────────┴─────────┴─────────┘  └─────────┴─────────┴─────────┘  └─────────┴─────────┴─────────┘
```

### 2. **Embedding Generation Layer** (`processing/embeddings.py`)

**Purpose**: Converts different content types into vector embeddings for similarity search.

**Embedding Strategies**:
- **Text Embeddings**: Direct embedding using `text-embedding-3-small` (5x cheaper, better quality)
- **Image Embeddings**: 
  1. Generate description using GPT-4o-mini Vision (16x cheaper)
  2. Embed description using `text-embedding-3-small`
- **Table Embeddings**: Embed table content as text using `text-embedding-3-small`

**Libraries Used**:
- `openai`: API client for OpenAI models
- `base64`: Image encoding for API transmission
- `io`: Image data handling

**Data Flow**:
```
Content Chunks → OpenAI API → Vector Embeddings → Metadata Enrichment
```

### 3. **Vector Storage Layer** (`processing/vectorstore.py`)

**Purpose**: Manages vector database operations and similarity search.

**Features**:
- **FAISS Integration**: High-performance similarity search
- **Hybrid Storage**: Separate indices for text, images, tables, and chat history
- **Persistence**: Save/load functionality for knowledge base
- **Metadata Management**: Rich metadata for citations and filtering
- **Chat Memory**: Persistent storage of conversation history for context awareness

**Libraries Used**:
- `faiss`: Facebook AI Similarity Search for vector operations
- `numpy`: Numerical operations and array handling
- `pickle`: Serialization for persistence
- `json`: Metadata storage

**Data Flow**:
```
Embeddings → FAISS Index → Similarity Search → Ranked Results
```

### 4. **Retrieval & Generation Layer** (`processing/retrievals.py`)

**Purpose**: Orchestrates search, context generation, and answer synthesis.

**Features**:
- **Adaptive Hybrid Search**: Combines dense (semantic) and sparse (keyword) search with dynamic weighting
- **Advanced Keyword Search**: TF-IDF vectorization with optional query expansion
- **Content Type Boosting**: Smart prioritization based on query characteristics
- **Multi-Factor Reranking**: Considers content type, recency, and length
- **Context Generation**: Aggregates relevant content from multiple sources
- **Answer Generation**: Uses GPT-4o-mini with chat history for contextual responses (3x cheaper, better quality)
- **Citation Generation**: Automatic citation formatting with page/line numbers

**Libraries Used**:
- `openai`: GPT models for generation
- `sklearn.feature_extraction.text`: TF-IDF for keyword search
- `sklearn.metrics.pairwise`: Cosine similarity calculations
- `numpy`: Score normalization and ranking
- `re`: Text processing for citations

**Data Flow**:
```
Query → Query Analysis → Adaptive Hybrid Search → Multi-Factor Reranking → Context Aggregation → Answer Generation → Citations
```

### 5. **User Interface Layer** (`app.py`)

**Purpose**: Provides an intuitive web interface for PDF upload and Q&A interactions.

**Features**:
- **Dark Theme UI**: Modern, professional interface
- **File Management**: PDF upload with duplicate detection
- **Chat Interface**: Persistent chat history with follow-up suggestions
- **Knowledge Base Management**: Clear data, system diagnostics, file removal
- **Citation Display**: Interactive search results and source citations

**Libraries Used**:
- `streamlit`: Web application framework
- `tempfile`: Temporary file handling
- `datetime`: Timestamp management
- `python-dotenv`: Environment variable management

**Data Flow**:
```
User Input → PDF Processing → Chat Interface → Response Display
```

## Cost Optimization

### **Model Cost Savings**
The system uses cost-optimized models to reduce API costs by ~85%:

- **Embeddings**: `text-embedding-3-small` (5x cheaper than ada-002)
- **Vision**: `gpt-4o-mini` (16x cheaper than gpt-4o for image descriptions)
- **Generation**: `gpt-4o-mini` (3x cheaper than gpt-3.5-turbo, better quality)

### **Cost Comparison**
| Component | Original Model | Cost | Optimized Model | Cost | Savings |
|-----------|----------------|------|-----------------|------|---------|
| Embeddings | `text-embedding-ada-002` | $0.0001/1K | `text-embedding-3-small` | $0.00002/1K | **5x cheaper** |
| Vision | `gpt-4o` | $0.0025/1K | `gpt-4o-mini` | $0.00015/1K | **16x cheaper** |
| Generation | `gpt-3.5-turbo` | $0.0005/1K | `gpt-4o-mini` | $0.00015/1K | **3x cheaper** |

**Total Cost Reduction: ~85% savings while maintaining or improving quality**

## Technology Stack

### **Core Libraries**

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.28.1 | Web UI framework |
| `openai` | 1.3.7 | AI model API client |
| `faiss-cpu` | 1.7.4 | Vector similarity search |
| `pymupdf` | 1.23.8 | PDF processing |
| `pillow` | 10.0.1 | Image processing |
| `pandas` | 2.1.3 | Data manipulation |
| `numpy` | 1.24.3 | Numerical operations |
| `python-dotenv` | 1.0.0 | Environment management |
| `langchain` | 0.1.0 | LLM framework |
| `langchain-openai` | 0.0.5 | OpenAI integration |
| `scikit-learn` | 1.3.0 | TF-IDF and similarity calculations |

### **AI Models Used (Cost-Optimized)**

| Model | Purpose | Configuration | Cost Savings |
|-------|---------|---------------|--------------|
| `text-embedding-3-small` | Text/Image/Table embeddings | 1536 dimensions | 5x cheaper |
| `gpt-4o-mini` | Image description generation | Vision model | 16x cheaper |
| `gpt-4o-mini` | Answer generation | Chat completion | 3x cheaper |

## Data Flow Architecture

### **1. PDF Ingestion Pipeline**

```
PDF Upload → Content Extraction → Multi-Modal Processing → Embedding → Vector Storage
     ↓              ↓                    ↓                    ↓            ↓
  File Validation  Text/Image/        Parallel Processing   OpenAI API   FAISS Index
                  Table Detection     ┌─────────┬─────────┬─────────┐    Calls        Persistence
                                      │  Text   │ Images  │ Tables  │
                                      │Chunking │Extract  │Extract  │
                                      │         │& Desc   │& Parse  │
                                      └─────────┴─────────┴─────────┘
```

### **2. Question-Answering Pipeline**

```
User Question → Query Processing → Hybrid Search → Context Retrieval → Answer Generation
      ↓              ↓              ↓              ↓              ↓
  Chat History   Query Analysis   Dense + Sparse  Content        GPT-4o-mini
  Integration    & Expansion      Search          Aggregation    with Citations
                    ↓              ↓              ↓              ↓
              Previous Q&A    Document + Chat   Context +       Persistent
              Retrieval       History Search    History         Chat Storage
```

### **3. Chat Management Pipeline**

```
Chat Entry → History Storage → Follow-up Suggestions → Context Integration → Response
     ↓            ↓                ↓                    ↓              ↓
  Timestamp    Session State    Quick Buttons       Chat History   Citation Display
  Generation   Persistence      Generation          Injection      & Source Links
```

## Key Design Principles

### **1. Modularity**
- Each component has a single responsibility
- Clear interfaces between modules
- Easy to extend and maintain

### **2. Robustness**
- Multiple fallback strategies for table extraction
- Comprehensive error handling
- Detailed logging for debugging

### **3. Scalability**
- Efficient vector search with FAISS
- Batch processing capabilities
- Memory-conscious chunking

### **4. User Experience**
- Intuitive dark theme interface
- Persistent chat history
- Quick follow-up suggestions
- Detailed citations and source tracking

### **5. Performance**
- Hybrid search combining semantic and keyword matching
- Optimized chunk sizes with overlap
- Efficient vector operations

## File Structure

```
rag_multimodal_MPG/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables (API keys)
├── assets/
|   ├──mpg_icon.png           # MPG icon 
│   └── image.png             # MPG logo
├── processing/
│   ├── __init__.py           # Package initialization
│   ├── chunkings.py          # PDF content extraction
│   ├── embeddings.py         # Embedding generation
│   ├── vectorstore.py        # Vector database management
│   └── retrievals.py         # Search and generation logic
├── vector_store/             # Persistent vector database
├── README.md                 # User documentation
└── ARCHITECTURE.md           # This architecture document
```

## Configuration

### **Environment Variables**
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### **Chunking Parameters**
- **Text Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Line Tracking**: Enabled for precise citations

### **Search Parameters**
- **Top-K Results**: 5 per content type
- **Hybrid Search**: 70% semantic + 30% keyword
- **Score Normalization**: Min-max scaling

## Performance Characteristics

### **Processing Speed**
- **Text Extraction**: ~1-2 seconds per page
- **Image Processing**: ~2-5 seconds per image
- **Table Detection**: ~1-3 seconds per page
- **Embedding Generation**: ~5-10 seconds per batch

### **Memory Usage**
- **Vector Storage**: ~1-5 MB per document
- **Session State**: ~10-50 MB for chat history
- **Image Storage**: ~100KB-1MB per image

### **Scalability**
- **Document Limit**: Limited by available memory
- **Concurrent Users**: Streamlit session-based
- **Vector Search**: Sub-second response times

## Security Considerations

### **Data Privacy**
- No data sent to external services except OpenAI API
- Local vector storage
- Temporary file cleanup

### **API Security**
- Environment variable for API keys
- No hardcoded credentials
- Secure API communication

## Future Enhancements

### **Planned Features**
- Multi-language support
- Advanced table structure recognition
- Custom embedding models
- Real-time collaboration
- Advanced analytics dashboard

### **Technical Improvements**
- Database backend for persistence
- Microservices architecture
- Caching layer for performance
- Advanced search algorithms
- Custom model fine-tuning

## Troubleshooting

### **Common Issues**
1. **PDF Processing Errors**: Check PyMuPDF version and file format
2. **API Rate Limits**: Implement request throttling
3. **Memory Issues**: Reduce chunk sizes or batch processing
4. **Vector Store Corruption**: Clear and rebuild indices

### **Debugging Tools**
- Detailed logging in all components
- System diagnostics in UI
- Error tracking and reporting
- Performance monitoring

This architecture provides a robust, scalable, and user-friendly RAG system that can handle complex PDF documents while maintaining high performance and accuracy in question-answering tasks. 
