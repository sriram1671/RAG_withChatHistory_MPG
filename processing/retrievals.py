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
        """Format search results as citations."""
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
        """Get context for question answering with citations and chat history."""
        # Perform hybrid search
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
        """Generate answer using OpenAI with context, citations, and chat history."""
        try:
            # Build conversation history
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. Always include citations. Use the conversation history to provide better, more contextual answers for follow-up questions."}
            ]
            
            # Add recent chat history (last 3 exchanges)
            if chat_history:
                recent_history = chat_history[-6:]  # Last 3 Q&A pairs
                for entry in recent_history:
                    messages.append({"role": "user", "content": entry['question']})
                    messages.append({"role": "assistant", "content": entry['answer']})
            
            # Add current context and question
            current_prompt = f"""Based on the following context from a document, answer the question. 
            Always include the relevant citations in your answer.

            Context:
            {context}

            Question: {query}

            Instructions:
            1. Answer the question based only on the provided context
            2. If the answer cannot be found in the context, say so
            3. Always include the relevant citations (e.g., [1], [2]) in your answer
            4. Be concise but comprehensive
            5. If this is a follow-up question, reference previous context when relevant

            Answer:"""
            
            messages.append({"role": "user", "content": current_prompt})

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            # Add citations at the end if not already included
            if citations and not any(f"[{i}]" in answer for i in range(1, 10)):
                answer += f"\n\nCitations: {citations}"
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I apologize, but I encountered an error while generating the answer. Please try again. Citations: {citations}" 