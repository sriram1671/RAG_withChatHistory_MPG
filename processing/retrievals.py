import openai
from typing import List, Dict, Any, Tuple
import re
from .vectorstore import FAISSVectorStore
from .embeddings import EmbeddingManager

class RetrievalManager:
    def __init__(self, openai_api_key: str, vector_store: FAISSVectorStore):
        self.embedding_manager = EmbeddingManager(openai_api_key)
        self.vector_store = vector_store
        self.client = openai.OpenAI(api_key=openai_api_key)
        # Cost-optimized generation model: 3x cheaper, better quality
        self.generation_model = "gpt-4o-mini"  # 3x cheaper than gpt-3.5-turbo
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for the query."""
        return self.embedding_manager.get_text_embedding(query)
    
    def keyword_search(self, query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        """Perform keyword-based search on chunks."""
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
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense embeddings and keyword search.
        alpha: weight for dense search (1-alpha for keyword search)
        """
        # Get dense search results
        query_embedding = self.get_query_embedding(query)
        dense_results = self.vector_store.search(query_embedding, k=k*2)  # Get more for reranking
        
        # Get keyword search results
        all_chunks = []
        all_chunks.extend(self.vector_store.text_metadata)
        all_chunks.extend(self.vector_store.image_metadata)
        all_chunks.extend(self.vector_store.table_metadata)
        all_chunks.extend(self.vector_store.chat_metadata)
        
        keyword_results = self.keyword_search(query, all_chunks, k=k*2)
        
        # Combine and rerank results
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
    
    def get_context_for_qa(self, query: str, chat_history: List[Dict[str, Any]] = None, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """Get context for question answering."""
        # Get relevant chunks
        results = self.hybrid_search(query, k=k)
        
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