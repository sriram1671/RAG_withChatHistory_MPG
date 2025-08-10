import openai
from typing import List, Dict, Any
import base64
import io
from PIL import Image
import numpy as np

class EmbeddingManager:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI text-embedding-ada-002."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting text embedding: {e}")
            return []
    
    def get_image_description(self, image_data: bytes) -> str:
        """Get description of image using OpenAI Vision model."""
        try:
            # Convert bytes to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail. Focus on any text, charts, diagrams, or important visual elements that might be relevant for document understanding."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting image description: {e}")
            # Return a generic description for images that can't be processed
            return "Image content (visual element from document)"
    
    def get_image_embedding(self, image_data: bytes) -> List[float]:
        """Get embedding for image using OpenAI CLIP model."""
        try:
            # Convert bytes to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=image_base64,
                encoding_format="base64"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting image embedding: {e}")
            return []
    
    def get_table_embedding(self, table_content: str) -> List[float]:
        """Get embedding for table content."""
        try:
            # Add context to table content for better embedding
            enhanced_content = f"Table data: {table_content}"
            return self.get_text_embedding(enhanced_content)
        except Exception as e:
            print(f"Error getting table embedding: {e}")
            return []
    
    def process_chunks(self, chunks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Process all chunks and add embeddings."""
        processed_chunks = {
            'text_chunks': [],
            'images': [],
            'tables': []
        }
        
        # Process text chunks
        print("Processing text chunks...")
        for chunk in chunks['text_chunks']:
            embedding = self.get_text_embedding(chunk['content'])
            if embedding:
                chunk['embedding'] = embedding
                processed_chunks['text_chunks'].append(chunk)
        
        # Process images
        print("Processing images...")
        for image in chunks['images']:
            # Get image description first
            description = self.get_image_description(image['content'])
            image['description'] = description
            
            # Get embedding for the description
            embedding = self.get_text_embedding(description)
            if embedding:
                image['embedding'] = embedding
                processed_chunks['images'].append(image)
        
        # Process tables
        print("Processing tables...")
        for table in chunks['tables']:
            embedding = self.get_table_embedding(table['content'])
            if embedding:
                table['embedding'] = embedding
                processed_chunks['tables'].append(table)
        
        print(f"Processed {len(processed_chunks['text_chunks'])} text chunks, "
              f"{len(processed_chunks['images'])} images, {len(processed_chunks['tables'])} tables")
        
        return processed_chunks 