import openai
from typing import List, Dict, Any, Tuple
import re
from .vectorstore import FAISSVectorStore
from .embeddings import EmbeddingManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RetrievalManager:
    def __init__(self, openai_api_key: str, vector_store: FAISSVectorStore):
        self.embedding_manager = EmbeddingManager(openai_api_key)
        self.vector_store = vector_store
        self.client = openai.OpenAI(api_key=openai_api_key)
        # Cost-optimized generation model: 3x cheaper, better quality
        self.generation_model = "gpt-4o-mini"  # 3x cheaper than gpt-3.5-turbo
        
        # Initialize TF-IDF for better keyword search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.95
        )
        self.tfidf_matrix = None
        self.tfidf_fitted = False
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for the query."""
        return self.embedding_manager.get_text_embedding(query)
    
    def expand_query(self, query: str) -> str:
        """Expand query with general synonyms and related terms."""
        # General query expansion - can be enhanced with WordNet or LLM
        expansions = {
            'increase': ['growth', 'rise', 'improvement', 'enhancement'],
            'decrease': ['decline', 'reduction', 'drop', 'fall'],
            'performance': ['results', 'achievement', 'outcome', 'success'],
            'data': ['information', 'figures', 'statistics', 'numbers'],
            'show': ['display', 'present', 'illustrate', 'demonstrate'],
            'explain': ['describe', 'clarify', 'elaborate', 'detail'],
            'compare': ['contrast', 'analyze', 'evaluate', 'assess'],
            'find': ['locate', 'identify', 'discover', 'search'],
            'list': ['enumerate', 'catalog', 'itemize', 'specify']
        }
        
        expanded_terms = []
        query_lower = query.lower()
        
        for term, synonyms in expansions.items():
            if term in query_lower:
                expanded_terms.extend(synonyms[:2])  # Add top 2 synonyms
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query
    
    def setup_tfidf(self):
        """Setup TF-IDF vectorizer with all content."""
        if self.tfidf_fitted:
            return
            
        all_texts = []
        all_chunks = []
        
        # Collect all text content
        for chunk in self.vector_store.text_metadata:
            all_texts.append(chunk['content'])
            all_chunks.append(chunk)
        
        for chunk in self.vector_store.image_metadata:
            all_texts.append(chunk['description'])
            all_chunks.append(chunk)
        
        for chunk in self.vector_store.table_metadata:
            all_texts.append(chunk['content'])
            all_chunks.append(chunk)
        
        if all_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            self.tfidf_fitted = True
    
    def advanced_keyword_search(self, query: str, k: int = 5, use_expansion: bool = True) -> List[Dict[str, Any]]:
        """Advanced keyword search using TF-IDF and cosine similarity."""
        if not self.tfidf_fitted:
            self.setup_tfidf()
        
        if not self.tfidf_fitted:
            return self.simple_keyword_search(query, k)
        
        # Optionally expand query
        search_query = self.expand_query(query) if use_expansion else query
        
        # Transform query to TF-IDF
        query_vector = self.tfidf_vectorizer.transform([search_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k*2]  # Get more for reranking
        
        results = []
        all_chunks = []
        all_chunks.extend(self.vector_store.text_metadata)
        all_chunks.extend(self.vector_store.image_metadata)
        all_chunks.extend(self.vector_store.table_metadata)
        
        for idx in top_indices:
            if idx < len(all_chunks) and similarities[idx] > 0:
                result = all_chunks[idx].copy()
                result['keyword_score'] = float(similarities[idx])
                results.append(result)
        
        return results[:k]
    
    def simple_keyword_search(self, query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword-based search on chunks (fallback)."""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        results = []
        
        for chunk in chunks:
            content = chunk.get('content', '').lower()
            description = chunk.get('description', '').lower()
            
            # Calculate keyword match score
            content_terms = set(re.findall(r'\b\w+\b', content))
            description_terms = set(re.findall(r'\b\w+\b', description))
            
            # Count matching terms
            content_matches = len(query_terms.intersection(content_terms))
            description_matches = len(query_terms.intersection(description_terms))
            
            # Calculate score (weighted combination)
            score = (content_matches * 2) + description_matches
            
            if score > 0:
                result = chunk.copy()
                result['keyword_score'] = score
                results.append(result)
        
        # Sort by keyword score and return top k
        results.sort(key=lambda x: x['keyword_score'], reverse=True)
        return results[:k]
    
    def calculate_content_type_boost(self, query: str, result: Dict[str, Any]) -> float:
        """Calculate content type specific boost based on query characteristics."""
        query_lower = query.lower()
        content_type = result.get('type', 'text')
        
        # Image-related queries
        image_keywords = ['image', 'picture', 'chart', 'graph', 'diagram', 'visual', 'figure', 'photo']
        if any(keyword in query_lower for keyword in image_keywords) and content_type == 'image':
            return 1.5
        
        # Table-related queries
        table_keywords = ['table', 'data', 'numbers', 'statistics', 'figures', 'values', 'list']
        if any(keyword in query_lower for keyword in table_keywords) and content_type == 'table':
            return 1.3
        
        # Quantitative queries (general)
        quantitative_keywords = ['amount', 'total', 'percentage', 'count', 'sum', 'average']
        if any(keyword in query_lower for keyword in quantitative_keywords) and content_type == 'table':
            return 1.2
        
        return 1.0
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using multiple factors."""
        for result in results:
            # Content type boost
            content_boost = self.calculate_content_type_boost(query, result)
            
            # Recency boost (for chat history)
            if result.get('type') == 'chat':
                recency_boost = 1.1  # Slight boost for recent conversations
            else:
                recency_boost = 1.0
            
            # Length boost (prefer longer, more detailed content)
            content_length = len(result.get('content', ''))
            length_boost = min(1.2, 1.0 + (content_length / 1000) * 0.2)
            
            # Apply boosts
            result['final_score'] = (
                result.get('combined_score', 0) * 
                content_boost * 
                recency_boost * 
                length_boost
            )
        
        # Sort by final score
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return results
    
    def adaptive_hybrid_search(self, query: str, k: int = 5, use_expansion: bool = True) -> List[Dict[str, Any]]:
        """
        Adaptive hybrid search that adjusts weights based on query characteristics.
        """
        # Analyze query type
        query_lower = query.lower()
        
        # Determine if query is more semantic or keyword-based
        semantic_indicators = ['what is', 'how does', 'explain', 'describe', 'why', 'compare']
        keyword_indicators = ['find', 'show', 'list', 'numbers', 'data', 'table']
        
        semantic_count = sum(1 for indicator in semantic_indicators if indicator in query_lower)
        keyword_count = sum(1 for indicator in keyword_indicators if indicator in query_lower)
        
        # Adaptive alpha based on query type
        if semantic_count > keyword_count:
            alpha = 0.8  # More weight on semantic search
        elif keyword_count > semantic_count:
            alpha = 0.4  # More weight on keyword search
        else:
            alpha = 0.6  # Balanced approach
        
        # Get dense search results
        query_embedding = self.get_query_embedding(query)
        dense_results = self.vector_store.search(query_embedding, k=k*2)
        
        # Get keyword search results
        all_chunks = []
        all_chunks.extend(self.vector_store.text_metadata)
        all_chunks.extend(self.vector_store.image_metadata)
        all_chunks.extend(self.vector_store.table_metadata)
        all_chunks.extend(self.vector_store.chat_metadata)
        
        keyword_results = self.advanced_keyword_search(query, k=k*2, use_expansion=use_expansion)
        
        # Combine results
        combined_results = {}
        
        # Add dense search results
        for result in dense_results:
            key = f"{result['type']}_{result.get('page', 0)}_{result.get('line_start', 0)}"
            combined_results[key] = {
                'result': result,
                'dense_score': result.get('score', 0),
                'keyword_score': 0
            }
        
        # Add keyword search results
        for result in keyword_results:
            key = f"{result['type']}_{result.get('page', 0)}_{result.get('line_start', 0)}"
            if key in combined_results:
                combined_results[key]['keyword_score'] = result.get('keyword_score', 0)
            else:
                combined_results[key] = {
                    'result': result,
                    'dense_score': 0,
                    'keyword_score': result.get('keyword_score', 0)
                }
        
        # Calculate combined scores with adaptive weights
        final_results = []
        for key, data in combined_results.items():
            combined_score = (alpha * data['dense_score']) + ((1 - alpha) * data['keyword_score'])
            result = data['result'].copy()
            result['combined_score'] = combined_score
            result['dense_score'] = data['dense_score']
            result['keyword_score'] = data['keyword_score']
            final_results.append(result)
        
        # Rerank results
        final_results = self.rerank_results(query, final_results)
        
        return final_results[:k]
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7, use_expansion: bool = True) -> List[Dict[str, Any]]:
        """
        Legacy hybrid search - now calls adaptive hybrid search.
        """
        return self.adaptive_hybrid_search(query, k, use_expansion)
    
    def simple_hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Simple hybrid search without query expansion - faster and more direct.
        """
        # Get dense search results
        query_embedding = self.get_query_embedding(query)
        dense_results = self.vector_store.search(query_embedding, k=k*2)
        
        # Get keyword search results (without expansion)
        all_chunks = []
        all_chunks.extend(self.vector_store.text_metadata)
        all_chunks.extend(self.vector_store.image_metadata)
        all_chunks.extend(self.vector_store.table_metadata)
        all_chunks.extend(self.vector_store.chat_metadata)
        
        keyword_results = self.advanced_keyword_search(query, k=k*2, use_expansion=False)
        
        # Combine results
        combined_results = {}
        
        # Add dense search results
        for result in dense_results:
            key = f"{result['type']}_{result.get('page', 0)}_{result.get('line_start', 0)}"
            combined_results[key] = {
                'result': result,
                'dense_score': result.get('score', 0),
                'keyword_score': 0
            }
        
        # Add keyword search results
        for result in keyword_results:
            key = f"{result['type']}_{result.get('page', 0)}_{result.get('line_start', 0)}"
            if key in combined_results:
                combined_results[key]['keyword_score'] = result.get('keyword_score', 0)
            else:
                combined_results[key] = {
                    'result': result,
                    'dense_score': 0,
                    'keyword_score': result.get('keyword_score', 0)
                }
        
        # Calculate combined scores
        final_results = []
        for key, data in combined_results.items():
            combined_score = (alpha * data['dense_score']) + ((1 - alpha) * data['keyword_score'])
            result = data['result'].copy()
            result['combined_score'] = combined_score
            result['dense_score'] = data['dense_score']
            result['keyword_score'] = data['keyword_score']
            final_results.append(result)
        
        # Sort by combined score and return top k
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return final_results[:k]
    
    def format_citations(self, results: List[Dict[str, Any]]) -> str:
        """Format citations for the results."""
        citations = []
        for i, result in enumerate(results, 1):
            if result['type'] == 'text':
                citation = f"[{i}] Page {result['page']}, Lines {result['line_start']}-{result['line_end']}"
            elif result['type'] == 'image':
                citation = f"[{i}] Page {result['page']}, Image {result['image_index']}"
            elif result['type'] == 'table':
                citation = f"[{i}] Page {result['page']}, Table {result['table_index']}"
            elif result['type'] == 'chat':
                citation = f"[{i}] Previous conversation: {result['timestamp']}"
            else:
                citation = f"[{i}] Page {result.get('page', 'Unknown')}"
            citations.append(citation)
        return "; ".join(citations)
    
    def get_context_for_qa(self, query: str, chat_history: List[Dict[str, Any]] = None, k: int = 5, use_expansion: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
        """Get context for question answering."""
        # Get relevant chunks
        results = self.adaptive_hybrid_search(query, k=k, use_expansion=use_expansion)
        
        # Build context string
        context_parts = []
        for i, result in enumerate(results, 1):
            if result['type'] == 'text':
                context_parts.append(f"[{i}] {result['content']}")
            elif result['type'] == 'image':
                context_parts.append(f"[{i}] Image: {result['description']}")
            elif result['type'] == 'table':
                context_parts.append(f"[{i}] Table: {result['content']}")
            elif result['type'] == 'chat':
                context_parts.append(f"[{i}] Previous Q&A: {result['content']}")
        
        context = "\n\n".join(context_parts)
        citations = self.format_citations(results)
        
        return context, results
    
    def answer_question(self, query: str, context: str, citations: str, chat_history: List[Dict[str, Any]] = None) -> str:
        """Generate answer using GPT-4o-mini (cost-optimized)."""
        try:
            # Build system message
            system_message = """You are a helpful AI assistant that answers questions based on provided context from PDF documents. 
            Always provide accurate answers based on the given context and include citations when possible.
            If the context doesn't contain enough information to answer the question, say so clearly."""
            
            # Build user message with context and chat history
            user_message = f"Question: {query}\n\nContext:\n{context}\n\nCitations: {citations}"
            
            # Add chat history if available
            if chat_history:
                chat_context = "\n\nPrevious conversation:\n"
                for entry in chat_history[-3:]:  # Last 3 exchanges
                    chat_context += f"Q: {entry['question']}\nA: {entry['answer']}\n"
                user_message += chat_context
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.generation_model,  # gpt-4o-mini instead of gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {e}" 