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
from sklearn.metrics.pairwise import cosine_similarity

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
    
    def generate_embedding(self, text):
        """Generate embeddings using OpenAI API with fallback"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text[:8000]
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            # Fallback to local model
            try:
                embedding = self.embedding_model.encode(text)
                return np.array(embedding)
            except Exception as e2:
                logger.error(f"Local embedding failed: {e2}")
                return None
    
    def semantic_search(self, query, top_k=5, grade_filter=None, topic_filter=None):
        """Perform semantic search with fallback methods"""
        try:
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Build queryset with filters
            queryset = Content.objects.filter(embedding__isnull=False)
            
            if grade_filter:
                queryset = queryset.filter(grade=grade_filter)
            if topic_filter:
                queryset = queryset.filter(topic__icontains=topic_filter)
            
            print(f"Searching in {queryset.count()} content items")
            
            if queryset.count() == 0:
                logger.warning("No content with embeddings found")
                return self._fallback_search(query, grade_filter, topic_filter)
            
            # Try pgvector first, fallback to manual calculation
            try:
                return self._pgvector_search(query_embedding, queryset, top_k)
            except Exception as e:
                logger.warning(f"pgvector search failed: {e}")
                return self._manual_similarity_search(query_embedding, queryset, top_k)
        
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return self._fallback_search(query, grade_filter, topic_filter)
    
    def _pgvector_search(self, query_embedding, queryset, top_k):
        """Use pgvector for similarity search with VectorField"""
        with connection.cursor() as cursor:
            query_vector = query_embedding.tolist()
            
            # Get IDs from queryset
            content_ids = list(queryset.values_list('id', flat=True))
            
            if not content_ids:
                return []
            
            placeholders = ','.join(['%s'] * len(content_ids))
            
            # Use proper table name and vector operations
            sql = f"""
            SELECT id, title, topic, grade, content_text, 
                   embedding <-> %s as distance
            FROM content_content 
            WHERE id IN ({placeholders}) AND embedding IS NOT NULL
            ORDER BY distance ASC LIMIT %s
            """
            
            params = [query_vector] + content_ids + [top_k]
            cursor.execute(sql, params)
            results = cursor.fetchall()
            
            return [
                {
                    'id': row[0],
                    'title': row[1],
                    'topic': row[2],
                    'grade': row[3],
                    'content': row[4],
                    'similarity': max(0, 1 - row[5])  # Ensure non-negative similarity
                }
                for row in results
            ]
    
    def _manual_similarity_search(self, query_embedding, queryset, top_k):
        """Manual cosine similarity calculation for VectorField"""
        results = []
        
        for content in queryset:
            try:
                if content.embedding:
                    # VectorField stores as numpy array or list
                    if hasattr(content.embedding, 'tolist'):
                        content_embedding = np.array(content.embedding.tolist())
                    else:
                        content_embedding = np.array(content.embedding)
                    
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        content_embedding.reshape(1, -1)
                    )[0][0]
                    
                    results.append({
                        'id': content.id,
                        'title': content.title,
                        'topic': content.topic,
                        'grade': content.grade,
                        'content': content.content_text,
                        'similarity': float(similarity)
                    })
            except Exception as e:
                logger.error(f"Error processing content {content.id}: {e}")
                continue
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def _fallback_search(self, query, grade_filter, topic_filter):
        """Fallback to keyword search when embeddings fail"""
        try:
            queryset = Content.objects.all()
            
            # Apply filters
            if grade_filter:
                queryset = queryset.filter(grade=grade_filter)
            if topic_filter:
                queryset = queryset.filter(topic__icontains=topic_filter)
            
            # Simple keyword search
            keywords = query.lower().split()
            for keyword in keywords:
                queryset = queryset.filter(
                    models.Q(title__icontains=keyword) |
                    models.Q(content_text__icontains=keyword)
                )
            
            results = []
            for content in queryset[:5]:  # Limit to top 5
                results.append({
                    'id': content.id,
                    'title': content.title,
                    'topic': content.topic,
                    'grade': content.grade,
                    'content': content.content_text,
                    'similarity': 0.5  # Default similarity for keyword search
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def generate_answer(self, question, persona_name='friendly', grade_filter=None, topic_filter=None):
        """Generate answer using RAG pipeline with better debugging"""
        try:
            print(f"Generating answer for: {question[:50]}...")
            
            # Debug: Check content availability
            total_content = Content.objects.count()
            content_with_embeddings = Content.objects.filter(embedding__isnull=False).count()
            
            print(f"Total content: {total_content}, With embeddings: {content_with_embeddings}")
            
            # Get persona
            try:
                persona = TutorPersona.objects.get(name=persona_name)
            except TutorPersona.DoesNotExist:
                # Create default persona
                persona = TutorPersona.objects.create(
                    name=persona_name,
                    system_prompt=self.default_personas.get(persona_name, self.default_personas['friendly']),
                    description=f"{persona_name.capitalize()} tutoring style"
                )
            
            # Retrieve relevant content
            relevant_content = self.semantic_search(
                question, 
                top_k=5, 
                grade_filter=grade_filter,
                topic_filter=topic_filter
            )
            
            print(f"Found {len(relevant_content)} relevant pieces of content")
            
            if not relevant_content:
                # If no content found, provide a more helpful response
                return {
                    'answer': self._generate_general_answer(question, persona.system_prompt),
                    'sources': [],
                    'confidence': 0.0,
                    'processing_time': 0
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
                'processing_time': 0
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': "I apologize, but I encountered an error while processing your question. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'processing_time': 0
            }
    
    def _generate_general_answer(self, question, system_prompt):
        """Generate a general answer when no specific content is found"""
        try:
            prompt = f"""
            {system_prompt}
            
            A student asked: "{question}"
            
            You don't have specific educational materials about this topic in your knowledge base, 
            but you can still provide a helpful general response. Provide a brief, educational answer 
            based on general knowledge and suggest how the student might learn more about this topic.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful educational tutor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating general answer: {e}")
            return "I don't have specific information about that topic in my knowledge base. Could you provide more context or ask about a different topic?"
    
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
            
            Context from educational materials:
            {context}
            
            Student's question: {question}
            
            Please provide a clear, educational response that helps the student understand the topic.
            """
            
            response = openai.chat.completions.create(
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