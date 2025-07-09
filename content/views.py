import json
from django.views import View
from django.http import JsonResponse
from django.db import transaction
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.db import models

from rag.llm_service_class import LLMService
from tutor.models import QuestionAnswer, TutorPersona
from .models import Content
from .serializers import ContentSerializer


@method_decorator(csrf_exempt, name='dispatch')
class UploadContentView(View):
    """Upload new textbook content with metadata"""
    
    def post(self, request):
        try:
            with transaction.atomic():
                # Extract file and metadata
                file = request.FILES.get('file')
                title = request.POST.get('title')
                topic = request.POST.get('topic')
                grade = request.POST.get('grade')
                
                # Read and process content
                content_text = file.read().decode('utf-8')
                
                # Create content record
                content = Content.objects.create(
                    title=title,
                    topic=topic,
                    grade=grade,
                    content_text=content_text
                )
                
                # Generate and store embeddings
                llm_service = LLMService()
                embedding = llm_service.generate_embedding(content_text)
                content.embedding = embedding
                content.save()
                
                return JsonResponse({
                    'message': 'Content uploaded successfully',
                    'content_id': content.id
                }, status=201)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)


@method_decorator(csrf_exempt, name='dispatch')
class AskQuestionView(View):
    """Handle student questions using RAG"""
    
    def post(self, request):
        try:
            # Parse JSON body
            data = json.loads(request.body)
            question = data.get('question')
            persona_name = data.get('persona', 'friendly')
            
            # Get persona
            persona = TutorPersona.objects.get(name=persona_name)
            
            # Perform semantic search
            llm_service = LLMService()
            relevant_content = llm_service.semantic_search(question)
            
            # Prepare context
            context = "\n\n".join([
                f"Title: {content.title}\nTopic: {content.topic}\nGrade: {content.grade}\nContent: {content.content_text[:500]}..."
                for content in relevant_content
            ])
            
            # Generate answer
            answer = llm_service.generate_answer(question, context, persona.system_prompt)
            
            # Save Q&A for logging
            qa = QuestionAnswer.objects.create(
                question=question,
                answer=answer,
                persona=persona,
                retrieved_content=[{
                    'id': content.id,
                    'title': content.title,
                    'topic': content.topic
                } for content in relevant_content]
            )
            
            return JsonResponse({
                'question': question,
                'answer': answer,
                'persona': persona_name,
                'relevant_sources': qa.retrieved_content
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)


class GetTopicsView(View):
    """Get topics filtered by grade"""
    
    def get(self, request):
        grade = request.GET.get('grade')
        
        queryset = Content.objects.values('topic').distinct()
        if grade:
            queryset = queryset.filter(grade=grade)
        
        topics = [item['topic'] for item in queryset]
        return JsonResponse({'topics': topics})


class GetMetricsView(View):
    """Get system metrics"""
    
    def get(self, request):
        total_content = Content.objects.count()
        total_questions = QuestionAnswer.objects.count()
        topics_count = Content.objects.values('topic').distinct().count()
        
        return JsonResponse({
            'total_content_files': total_content,
            'total_questions_answered': total_questions,
            'topics_covered': topics_count,
            'average_rating': QuestionAnswer.objects.filter(
                rating__isnull=False
            ).aggregate(avg_rating=models.Avg('rating'))['avg_rating']
        })


@method_decorator(csrf_exempt, name='dispatch')
class NaturalLanguageQueryView(View):
    """Handle natural language database queries"""
    
    def post(self, request):
        try:
            # Parse JSON body
            data = json.loads(request.body)
            query = data.get('query')
            llm_service = LLMService()
            
            # Convert to SQL
            sql_query = llm_service.natural_language_to_sql(query)
            
            # Execute safely (add validation)
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                results = cursor.fetchall()
            
            return JsonResponse({
                'query': query,
                'sql_generated': sql_query,
                'results': results
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)