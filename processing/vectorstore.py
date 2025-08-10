import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
import json

class FAISSVectorStore:
    def __init__(self, dimension: int = 1536):  # OpenAI ada-002 embedding dimension
        self.dimension = dimension
        self.text_index = None
        self.image_index = None
        self.table_index = None
        self.chat_index = None
        self.text_metadata = []
        self.image_metadata = []
        self.table_metadata = []
        self.chat_metadata = []
        self.is_initialized = False
    
    def initialize_indexes(self):
        """Initialize FAISS indexes for different content types."""
        # Initialize text index
        self.text_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Initialize image index
        self.image_index = faiss.IndexFlatIP(self.dimension)
        
        # Initialize table index
        self.table_index = faiss.IndexFlatIP(self.dimension)
        
        # Initialize chat index
        self.chat_index = faiss.IndexFlatIP(self.dimension)
        
        self.is_initialized = True
        print("FAISS indexes initialized")
    
    def add_text_chunks(self, chunks: List[Dict[str, Any]]):
        """Add text chunks to the vector store."""
        if not self.is_initialized:
            self.initialize_indexes()
        
        if not chunks:
            return
        
        # Prepare embeddings and metadata
        embeddings = []
        metadata_list = []
        
        for chunk in chunks:
            if 'embedding' in chunk and chunk['embedding']:
                embeddings.append(chunk['embedding'])
                metadata_list.append({
                    'type': 'text',
                    'content': chunk['content'],
                    'page': chunk['page'],
                    'line_start': chunk['line_start'],
                    'line_end': chunk['line_end'],
                    'metadata': chunk['metadata']
                })
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.text_index.add(embeddings_array)
            self.text_metadata.extend(metadata_list)
            print(f"Added {len(embeddings)} text chunks to vector store")
    
    def add_images(self, images: List[Dict[str, Any]]):
        """Add images to the vector store."""
        if not self.is_initialized:
            self.initialize_indexes()
        
        if not images:
            return
        
        # Prepare embeddings and metadata
        embeddings = []
        metadata_list = []
        
        for image in images:
            if 'embedding' in image and image['embedding']:
                embeddings.append(image['embedding'])
                metadata_list.append({
                    'type': 'image',
                    'description': image['description'],
                    'page': image['page'],
                    'image_index': image['image_index'],
                    'position': image['position'],
                    'metadata': image['metadata']
                })
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.image_index.add(embeddings_array)
            self.image_metadata.extend(metadata_list)
            print(f"Added {len(embeddings)} images to vector store")
    
    def add_tables(self, tables: List[Dict[str, Any]]):
        """Add tables to the vector store."""
        if not self.is_initialized:
            self.initialize_indexes()
        
        if not tables:
            return
        
        # Prepare embeddings and metadata
        embeddings = []
        metadata_list = []
        
        for table in tables:
            if 'embedding' in table and table['embedding']:
                embeddings.append(table['embedding'])
                metadata_list.append({
                    'type': 'table',
                    'content': table['content'],
                    'page': table['page'],
                    'table_index': table['table_index'],
                    'table_data': table['table_data'],
                    'metadata': table['metadata']
                })
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.table_index.add(embeddings_array)
            self.table_metadata.extend(metadata_list)
            print(f"Added {len(embeddings)} tables to vector store")
    
    def add_chat_chunks(self, chat_chunks: List[Dict[str, Any]]):
        """Add chat history chunks to the vector store."""
        if not self.is_initialized:
            self.initialize_indexes()
        
        if not chat_chunks:
            return
        
        # Prepare embeddings and metadata
        embeddings = []
        metadata_list = []
        
        for chat in chat_chunks:
            if 'embedding' in chat and chat['embedding']:
                embeddings.append(chat['embedding'])
                metadata_list.append({
                    'type': 'chat',
                    'content': chat['content'],
                    'question': chat['question'],
                    'answer': chat['answer'],
                    'timestamp': chat['timestamp'],
                    'metadata': chat['metadata']
                })
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.chat_index.add(embeddings_array)
            self.chat_metadata.extend(metadata_list)
            print(f"Added {len(embeddings)} chat chunks to vector store")
    
    def search(self, query_embedding: List[float], k: int = 5, content_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search across all content types and return top k results."""
        if not self.is_initialized:
            return []
        
        if content_types is None:
            content_types = ['text', 'image', 'table', 'chat']
        
        results = []
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search text chunks
        if 'text' in content_types and self.text_index and self.text_index.ntotal > 0:
            text_scores, text_indices = self.text_index.search(query_array, min(k, self.text_index.ntotal))
            for i, (score, idx) in enumerate(zip(text_scores[0], text_indices[0])):
                if idx < len(self.text_metadata):
                    result = self.text_metadata[idx].copy()
                    result['score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
        
        # Search images
        if 'image' in content_types and self.image_index and self.image_index.ntotal > 0:
            image_scores, image_indices = self.image_index.search(query_array, min(k, self.image_index.ntotal))
            for i, (score, idx) in enumerate(zip(image_scores[0], image_indices[0])):
                if idx < len(self.image_metadata):
                    result = self.image_metadata[idx].copy()
                    result['score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
        
        # Search tables
        if 'table' in content_types and self.table_index and self.table_index.ntotal > 0:
            table_scores, table_indices = self.table_index.search(query_array, min(k, self.table_index.ntotal))
            for i, (score, idx) in enumerate(zip(table_scores[0], table_indices[0])):
                if idx < len(self.table_metadata):
                    result = self.table_metadata[idx].copy()
                    result['score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
        
        # Search chat history
        if 'chat' in content_types and self.chat_index and self.chat_index.ntotal > 0:
            chat_scores, chat_indices = self.chat_index.search(query_array, min(k, self.chat_index.ntotal))
            for i, (score, idx) in enumerate(zip(chat_scores[0], chat_indices[0])):
                if idx < len(self.chat_metadata):
                    result = self.chat_metadata[idx].copy()
                    result['score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def save(self, directory: str):
        """Save the vector store to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save indexes
        if self.text_index:
            faiss.write_index(self.text_index, os.path.join(directory, 'text_index.faiss'))
        if self.image_index:
            faiss.write_index(self.image_index, os.path.join(directory, 'image_index.faiss'))
        if self.table_index:
            faiss.write_index(self.table_index, os.path.join(directory, 'table_index.faiss'))
        if self.chat_index:
            faiss.write_index(self.chat_index, os.path.join(directory, 'chat_index.faiss'))
        
        # Save metadata
        with open(os.path.join(directory, 'text_metadata.pkl'), 'wb') as f:
            pickle.dump(self.text_metadata, f)
        with open(os.path.join(directory, 'image_metadata.pkl'), 'wb') as f:
            pickle.dump(self.image_metadata, f)
        with open(os.path.join(directory, 'table_metadata.pkl'), 'wb') as f:
            pickle.dump(self.table_metadata, f)
        with open(os.path.join(directory, 'chat_metadata.pkl'), 'wb') as f:
            pickle.dump(self.chat_metadata, f)
        
        print(f"Vector store saved to {directory}")
    
    def load(self, directory: str):
        """Load the vector store from disk."""
        try:
            # Load indexes
            text_index_path = os.path.join(directory, 'text_index.faiss')
            image_index_path = os.path.join(directory, 'image_index.faiss')
            table_index_path = os.path.join(directory, 'table_index.faiss')
            chat_index_path = os.path.join(directory, 'chat_index.faiss')
            
            if os.path.exists(text_index_path):
                self.text_index = faiss.read_index(text_index_path)
            if os.path.exists(image_index_path):
                self.image_index = faiss.read_index(image_index_path)
            if os.path.exists(table_index_path):
                self.table_index = faiss.read_index(table_index_path)
            if os.path.exists(chat_index_path):
                self.chat_index = faiss.read_index(chat_index_path)
            
            # Load metadata
            with open(os.path.join(directory, 'text_metadata.pkl'), 'rb') as f:
                self.text_metadata = pickle.load(f)
            with open(os.path.join(directory, 'image_metadata.pkl'), 'rb') as f:
                self.image_metadata = pickle.load(f)
            with open(os.path.join(directory, 'table_metadata.pkl'), 'rb') as f:
                self.table_metadata = pickle.load(f)
            with open(os.path.join(directory, 'chat_metadata.pkl'), 'rb') as f:
                self.chat_metadata = pickle.load(f)
            
            self.is_initialized = True
            print(f"Vector store loaded from {directory}")
            print(f"Loaded {len(self.text_metadata)} text chunks, {len(self.image_metadata)} images, {len(self.table_metadata)} tables, {len(self.chat_metadata)} chat chunks")
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.is_initialized = False
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the vector store."""
        stats = {
            'text_chunks': len(self.text_metadata),
            'images': len(self.image_metadata),
            'tables': len(self.table_metadata),
            'chat_chunks': len(self.chat_metadata),
            'total_items': len(self.text_metadata) + len(self.image_metadata) + len(self.table_metadata) + len(self.chat_metadata)
        }
        return stats
    
    def clear_all(self):
        """Clear all data from the vector store."""
        # Reinitialize empty indexes
        self.initialize_indexes()
        
        # Clear metadata
        self.text_metadata = []
        self.image_metadata = []
        self.table_metadata = []
        self.chat_metadata = []
        
        # Save empty state
        self.save("vector_store")
        print("Vector store cleared successfully") 