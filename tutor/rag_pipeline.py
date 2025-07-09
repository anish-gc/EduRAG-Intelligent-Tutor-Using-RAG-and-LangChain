# apps/tutor/rag_pipeline.py
import openai
import numpy as np
from django.conf import settings
from django.db import connection
from sentence_transformers import SentenceTransformer
import logging
from django.db import models
from content.models import Content
from tutor.models import QuestionAnswer, TutorPersona

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.default_personas = {
            'friendly': "You are a friendly and encouraging tutor who makes learning fun and accessible.",
            'strict': "You are a strict but fair tutor who emphasizes discipline and accuracy.",
            'humorous': "You are a humorous tutor who uses jokes and fun examples to make learning enjoyable.",
            'encouraging': "You are an encouraging tutor who builds confidence and celebrates progress."
        }
    
    def initialize_personas(self):
        """Initialize default tutor personas"""
        for name, prompt in self.default_personas.items():
            TutorPersona.objects.get_or_create(
                name=name,
                defaults={
                    'system_prompt': prompt,
                    'description': f'{name.capitalize()} tutoring style'
                }
            )
    
    def generate_embedding(self, text):
        """Generate embeddings using OpenAI API"""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text[:8000]  # Limit text length
            )
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def process_content(self, content_obj):
        """Process uploaded content and generate embeddings"""
        try:
            # Chunk content into smaller pieces for better retrieval
            chunks = self._chunk_text(content_obj.content_text)
            
            for i, chunk in enumerate(chunks):
                embedding = self.generate_embedding(chunk)
                if embedding is not None:
                    # Create separate content records for each chunk
                    Content.objects.create(
                        title=f"{content_obj.title} - Part {i+1}",
                        topic=content_obj.topic,
                        grade=content_obj.grade,
                        content_text=chunk,
                        embedding=embedding.tolist()
                    )
            
            return True
        except Exception as e:
            logger.error(f"Error processing content: {e}")
            return False
    
    def _chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:  # At least half the chunk
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def semantic_search(self, query, top_k=5, grade_filter=None, topic_filter=None):
        """Perform semantic search using cosine similarity"""
        try:
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                return []
            
            # Build query with filters
            queryset = Content.objects.filter(embedding__isnull=False)
            
            if grade_filter:
                queryset = queryset.filter(grade=grade_filter)
            if topic_filter:
                queryset = queryset.filter(topic__icontains=topic_filter)
            
            # Use raw SQL for vector similarity (pgvector)
            with connection.cursor() as cursor:
                query_vector = query_embedding.tolist()
                
                sql = """
                SELECT id, title, topic, grade, content_text, 
                       (embedding <-> %s::vector) as distance
                FROM content 
                WHERE embedding IS NOT NULL
                """
                params = [query_vector]
                
                if grade_filter:
                    sql += " AND grade = %s"
                    params.append(grade_filter)
                
                if topic_filter:
                    sql += " AND topic ILIKE %s"
                    params.append(f"%{topic_filter}%")
                
                sql += " ORDER BY distance ASC LIMIT %s"
                params.append(top_k)
                
                cursor.execute(sql, params)
                results = cursor.fetchall()
                
                return [
                    {
                        'id': row[0],
                        'title': row[1],
                        'topic': row[2],
                        'grade': row[3],
                        'content': row[4],
                        'similarity': 1 - row[5]  # Convert distance to similarity
                    }
                    for row in results
                ]
        
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def generate_answer(self, question, persona_name='friendly', grade_filter=None, topic_filter=None):
        """Generate answer using RAG pipeline"""
        try:
            # Get persona
            persona = TutorPersona.objects.get(name=persona_name)
            
            # Retrieve relevant content
            relevant_content = self.semantic_search(
                question, 
                top_k=5, 
                grade_filter=grade_filter,
                topic_filter=topic_filter
            )
            
            if not relevant_content:
                return {
                    'answer': "I don't have specific information about that topic in my knowledge base. Could you provide more context or ask about a different topic?",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Prepare context from retrieved content
            context = self._prepare_context(relevant_content)
            
            # Generate answer using LLM
            answer = self._call_llm(question, context, persona.system_prompt)
            
            # Calculate confidence based on similarity scores
            confidence = np.mean([content['similarity'] for content in relevant_content])
            
            return {
                'answer': answer,
                'sources': relevant_content[:3],  # Top 3 sources
                'confidence': float(confidence),
                'persona': persona_name
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': "I apologize, but I encountered an error while processing your question. Please try again.",
                'sources': [],
                'confidence': 0.0
            }
    
    def _prepare_context(self, relevant_content):
        """Prepare context from relevant content"""
        context_parts = []
        
        for content in relevant_content:
            context_parts.append(
                f"Source: {content['title']} (Grade {content['grade']}, Topic: {content['topic']})\n"
                f"Content: {content['content'][:500]}...\n"
                f"Relevance: {content['similarity']:.2f}\n"
            )
        
        return "\n".join(context_parts)
    
    def _call_llm(self, question, context, system_prompt):
        """Call OpenAI API to generate answer"""
        try:
            full_prompt = f"""
            {system_prompt}
            
            You are helping a student with their question. Use the following context to provide a helpful, accurate answer.
            If the context doesn't contain enough information, say so and provide general guidance.
            
            Context from educational materials:
            {context}
            
            Student's question: {question}
            
            Please provide a clear, educational response that helps the student understand the topic.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an educational tutor."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again later."
    
    def natural_language_to_sql(self, nl_query):
        """Convert natural language query to SQL"""
        try:
            schema_info = """
            Database Schema:
            Table: content
            Columns: id (integer), title (text), topic (text), grade (text), content_text (text), created_at (timestamp)
            
            Sample data:
            - Grades: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
            - Topics: Mathematics, Science, English, History, Geography, Physics, Chemistry, Biology
            """
            
            prompt = f"""
            Given this database schema:
            {schema_info}
            
            Convert this natural language query to a safe SQL SELECT statement:
            "{nl_query}"
            
            Rules:
            1. Only use SELECT statements
            2. Use proper PostgreSQL syntax
            3. Return only the SQL query, no explanation
            4. Use appropriate WHERE clauses for filtering
            5. Include LIMIT clause if not specified (default 10)
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Basic safety check
            if not sql_query.upper().startswith('SELECT'):
                raise ValueError("Generated query is not a SELECT statement")
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error converting NL to SQL: {e}")
            return None
    
    def execute_nl_query(self, nl_query):
        """Execute natural language query and return results"""
        try:
            sql_query = self.natural_language_to_sql(nl_query)
            
            if not sql_query:
                return {
                    'error': 'Could not convert query to SQL',
                    'results': []
                }
            
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                
                # Convert to list of dictionaries
                formatted_results = [
                    dict(zip(columns, row)) for row in results
                ]
                
                return {
                    'sql_query': sql_query,
                    'results': formatted_results,
                    'count': len(formatted_results)
                }
                
        except Exception as e:
            logger.error(f"Error executing NL query: {e}")
            return {
                'error': str(e),
                'results': []
            }
    
    def log_interaction(self, question, answer, persona_name, sources):
        """Log question-answer interaction"""
        try:
            persona = TutorPersona.objects.get(name=persona_name)
            
            QuestionAnswer.objects.create(
                question=question,
                answer=answer,
                persona=persona,
                retrieved_content=sources
            )
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
            return {
                'total_content': Content.objects.count(),
                'total_questions': QuestionAnswer.objects.count(),
                'topics_covered': Content.objects.values('topic').distinct().count(),
                'grades_covered': Content.objects.values('grade').distinct().count(),
                'average_rating': QuestionAnswer.objects.filter(
                    rating__isnull=False
                ).aggregate(avg=models.Avg('rating'))['avg'] or 0.0,
                'top_topics': list(
                    Content.objects.values('topic')
                    .annotate(count=models.Count('id'))
                    .order_by('-count')[:5]
                )
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}