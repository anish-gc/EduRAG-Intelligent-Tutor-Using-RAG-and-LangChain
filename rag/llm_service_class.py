import openai
import numpy as np
from django.conf import settings
from django.db import connection
from sentence_transformers import SentenceTransformer


class LLMService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text):
        """Generates embedding for text content"""
        response = openai.embeddings.create(model="text-embedding-ada-002", input=text)
        return response["data"][0]["embedding"]

    def semantic_search(self, query, limit=5):
        """performs semantic search using pgvector"""
        from content.models import Content

        query_embedding = self.generate_embedding(query)
        # used pgvector for similarity search
        similar_content = Content.objects.filter(embedding__isnull=False).order_by(
            Content.embedding.cosine_distance(query_embedding)
        )[:limit]
        return similar_content

    def generate_answer(self, question, context, persona_prompt):
        """generate answer using RAG"""
        system_prompt = f"""
        {persona_prompt}
        
        You are an educational tutor. Use the following context to answer the student's question.
        If the context doesn't contain relevant information, politely say so and provide general guidance.
        
        Context:
        {context}
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    
    def natural_language_to_sql(self, query):
        """Convert natural language to SQL query"""
        schema_info = """
        Database Schema:
        - content table: id, title, topic, grade, content_text, created_at
        - Available grades: 1-12
        - Available topics: Math, Science, English, History, etc.
        
        """
        prompt = f"""
        {schema_info}
        
        Convert this natural language query to SQL:
        "{query}"
        
        Return only the SQL query, no explanations.
        """
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()