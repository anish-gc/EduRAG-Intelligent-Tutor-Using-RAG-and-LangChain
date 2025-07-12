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



# Enhanced metrics view that matches your frontend
class GetMetricsView(View):
    """Get comprehensive system metrics"""
    
    def get(self, request):
        try:
            # Basic counts
            total_content = Content.objects.count()
            total_questions = QuestionAnswer.objects.count()
            topics_count = Content.objects.values('topic').distinct().count()
            grades_count = Content.objects.values('grade').distinct().count()
            
            # Average rating
            avg_rating = QuestionAnswer.objects.filter(
                rating__isnull=False
            ).aggregate(avg_rating=models.Avg('rating'))['avg_rating'] or 0.0
            
            # Most popular topics
            popular_topics = list(
                Content.objects.values('topic')
                .annotate(count=models.Count('id'))
                .order_by('-count')[:5]
            )
            
            # Recent activity
            recent_questions = QuestionAnswer.objects.order_by('-created_at')[:5]
            
            # Persona usage statistics
            persona_stats = list(
                QuestionAnswer.objects.values('persona__name')
                .annotate(count=models.Count('id'))
                .order_by('-count')
            )
            
            return JsonResponse({
                'total_content_files': total_content,
                'total_questions_answered': total_questions,
                'topics_covered': topics_count,
                'grades_covered': grades_count,
                'average_rating': round(avg_rating, 2),
                'popular_topics': popular_topics,
                'persona_usage': persona_stats,
                'recent_activity': [
                    {
                        'id': qa.id,
                        'question': qa.question[:100] + '...' if len(qa.question) > 100 else qa.question,
                        'persona': qa.persona.name,
                        'rating': qa.rating,
                        'created_at': qa.created_at.isoformat()
                    }
                    for qa in recent_questions
                ],
                'system_health': {
                    'content_with_embeddings': Content.objects.filter(embedding__isnull=False).count(),
                    'content_without_embeddings': Content.objects.filter(embedding__isnull=True).count(),
                    'avg_response_quality': avg_rating
                }
            })
            
        except Exception as e:
            logger.error(f"Error in GetMetricsView: {str(e)}")
            return JsonResponse({
                'error': 'Failed to load metrics'
            }, status=500)

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
        
class ContentListView(View):
    """List all content with filtering and pagination"""
    
    def get(self, request):
        try:
            # Pagination parameters
            page = int(request.GET.get('page', 1))
            page_size = int(request.GET.get('page_size', 20))
            
            # Filter parameters
            topic_filter = request.GET.get('topic')
            grade_filter = request.GET.get('grade')
            search = request.GET.get('search', '').strip()
            
            # Build queryset
            queryset = Content.objects.all()
            
            if topic_filter:
                queryset = queryset.filter(topic__icontains=topic_filter)
            
            if grade_filter:
                queryset = queryset.filter(grade=grade_filter)
            
            if search:
                queryset = queryset.filter(
                    models.Q(title__icontains=search) |
                    models.Q(content_text__icontains=search)
                )
            
            # Pagination
            total_count = queryset.count()
            offset = (page - 1) * page_size
            content_list = queryset.order_by('-created_at')[offset:offset + page_size]
            
            return JsonResponse({
                'content': [
                    {
                        'id': content.id,
                        'title': content.title,
                        'topic': content.topic,
                        'grade': content.grade,
                        'content_preview': content.content_text[:200] + '...' if len(content.content_text) > 200 else content.content_text,
                        'has_embedding': content.embedding is not None,
                        'created_at': content.created_at.isoformat(),
                        'updated_at': content.updated_at.isoformat()
                    }
                    for content in content_list
                ],
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': total_count,
                    'total_pages': (total_count + page_size - 1) // page_size
                },
                'filters': {
                    'topic': topic_filter,
                    'grade': grade_filter,
                    'search': search
                }
            })
            
        except Exception as e:
            logger.error(f"Error in ContentListView: {str(e)}")
            return JsonResponse({
                'error': 'Failed to load content'
            }, status=500)

# class ContentDetailView(View):
#     """Get detailed view of specific content"""
    
#     def get(self, request, content_id):
#         try:
#             content = Content.objects.get(id=content_id)
            
#             return JsonResponse({
#                 'id': content.id,
#                 'title': content.title,
#                 'topic': content.topic,
#                 'grade': content.grade,
#                 'content_text': content.content_text,
#                 'file_path': content.file_path,
#                 'has_embedding': content.embedding is not None,
#                 'embedding_dimensions': len(content.embedding) if content.embedding else 0,
#                 'created_at': content.created_at.iso        