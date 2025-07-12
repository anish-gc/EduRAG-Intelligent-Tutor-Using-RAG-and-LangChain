from django.views import View
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json

from tutor.rag_pipeline import RAGPipeline
from .models import TutorPersona, QuestionAnswer










# tutor/views.py
import json
import logging
from django.views import View
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.db import models
from django.core.exceptions import ValidationError

from content.models import Content
from .models import TutorPersona, QuestionAnswer

logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class AskQuestionView(View):
    """Handle student questions using RAG pipeline"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag_pipeline = RAGPipeline()
    
    def post(self, request):
        try:
            # Parse JSON body
            data = json.loads(request.body)
            # Validate required fields
            question = data.get('question', '').strip()
            if not question:
                return JsonResponse({
                    'error': 'Question is required'
                }, status=400)
            
            # Validate question length
            if len(question) > 500:
                return JsonResponse({
                    'error': 'Question is too long. Maximum 500 characters.'
                }, status=400)
            
            # Extract optional parameters
            persona_name = data.get('persona', 'friendly')
            grade_filter = data.get('grade', None)
            topic_filter = data.get('topic', None)
            
            # Validate persona exists
            try:
                persona = TutorPersona.objects.filter(name=persona_name).first()
            except TutorPersona.DoesNotExist:
                # Create default persona if not exists
                persona = TutorPersona.objects.create(
                    name='friendly',
                    system_prompt="You are a friendly and encouraging tutor who makes learning fun and accessible.",
                    description="Friendly tutoring style"
                )
                persona_name = 'friendly'
            
            
            # Validate grade if provided
            if grade_filter:
                try:
                    grade_int = int(grade_filter)
                    if grade_int < 1 or grade_int > 12:
                        return JsonResponse({
                            'error': 'Grade must be between 1 and 12'
                        }, status=400)
                except ValueError:
                    return JsonResponse({
                        'error': 'Invalid grade format'
                    }, status=400)
            
            # Log the incoming request
            logger.info(f"Question received: {question[:100]}... | Persona: {persona_name} | Grade: {grade_filter} | Topic: {topic_filter}")
            
            # Generate answer using RAG pipeline
            result = self.rag_pipeline.generate_answer(
                question=question,
                persona_name=persona_name,
                grade_filter=grade_filter,
                topic_filter=topic_filter
            )
            
            # Log the interaction
            self.rag_pipeline.log_interaction(
                question=question,
                answer=result['answer'],
                persona_name=persona_name,
                sources=result.get('sources', [])
            )
            
            # Format response for frontend
            response_data = {
                'question': question,
                'answer': result['answer'],
                'persona': persona_name,
                'confidence': result.get('confidence', 0.0),
                'relevant_sources': self._format_sources(result.get('sources', [])),
                'metadata': {
                    'grade_filter': grade_filter,
                    'topic_filter': topic_filter,
                    'sources_count': len(result.get('sources', [])),
                    'processing_time': result.get('processing_time', 0)
                }
            }
            print("respisne data is", response_data)
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON format'
            }, status=400)
        
        except Exception as e:
            logger.error(f"Error in AskQuestionView: {str(e)}")
            return JsonResponse({
                'error': 'An unexpected error occurred. Please try again.',
                'details': str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }, status=500)
    
    def _format_sources(self, sources):
        """Format sources for frontend display"""
        formatted_sources = []
        
        for source in sources:
            formatted_source = {
                'id': source.get('id'),
                'title': source.get('title', 'Unknown'),
                'topic': source.get('topic', 'General'),
                'grade': source.get('grade', 'N/A'),
                'similarity': round(source.get('similarity', 0.0), 3),
                'excerpt': source.get('content', '')[:200] + '...' if len(source.get('content', '')) > 200 else source.get('content', '')
            }
            formatted_sources.append(formatted_source)
        
        return formatted_sources
    
    def get(self, request):
        """Handle GET request - return available options"""
        try:
            # Get available personas
            personas = TutorPersona.objects.all()
            
            # Get available topics and grades
            topics = Content.objects.values_list('topic', flat=True).distinct()
            grades = Content.objects.values_list('grade', flat=True).distinct()
            
            return JsonResponse({
                'available_personas': [
                    {
                        'name': persona.name,
                        'description': persona.description
                    }
                    for persona in personas
                ],
                'available_topics': sorted(list(topics)),
                'available_grades': sorted([int(g) for g in grades if g.isdigit()]),
                'usage_guidelines': {
                    'max_question_length': 500,
                    'supported_methods': ['POST'],
                    'required_fields': ['question'],
                    'optional_fields': ['persona', 'grade', 'topic']
                }
            })
            
        except Exception as e:
            logger.error(f"Error in AskQuestionView GET: {str(e)}")
            return JsonResponse({
                'error': 'Failed to load options'
            }, status=500)


# Additional view for rating answers (referenced in your frontend)
@method_decorator(csrf_exempt, name='dispatch')
class RateAnswerView(View):
    """Rate a previous answer"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            qa_id = data.get('qa_id')
            rating = data.get('rating')
            
            # Validate rating
            if not isinstance(rating, int) or rating < 1 or rating > 5:
                return JsonResponse({
                    'error': 'Rating must be an integer between 1 and 5'
                }, status=400)
            
            # Update the rating
            qa = QuestionAnswer.objects.get(id=qa_id)
            qa.rating = rating
            qa.save()
            
            return JsonResponse({
                'message': 'Rating saved successfully',
                'qa_id': qa_id,
                'rating': rating
            })
            
        except QuestionAnswer.DoesNotExist:
            return JsonResponse({
                'error': 'Question-Answer not found'
            }, status=404)
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON format'
            }, status=400)
        except Exception as e:
            logger.error(f"Error in RateAnswerView: {str(e)}")
            return JsonResponse({
                'error': 'Failed to save rating'
            }, status=500)




# Utility view for getting available topics (with filtering)
class GetTopicsView(View):
    """Get topics with optional filtering"""
    
    def get(self, request):
        try:
            grade = request.GET.get('grade')
            search = request.GET.get('search', '').strip()
            
            # Base queryset
            queryset = Content.objects.values('topic', 'grade').distinct()
            
            # Apply filters
            if grade:
                queryset = queryset.filter(grade=grade)
            
            if search:
                queryset = queryset.filter(topic__icontains=search)
            
            # Group by topic and include grade info
            topics_data = {}
            for item in queryset:
                topic = item['topic']
                grade = item['grade']
                
                if topic not in topics_data:
                    topics_data[topic] = {
                        'name': topic,
                        'grades': [],
                        'content_count': 0
                    }
                
                topics_data[topic]['grades'].append(grade)
            
            # Add content counts
            for topic_name in topics_data:
                topics_data[topic_name]['content_count'] = Content.objects.filter(
                    topic=topic_name
                ).count()
                topics_data[topic_name]['grades'] = sorted(list(set(topics_data[topic_name]['grades'])))
            
            # Convert to list and sort
            topics_list = sorted(list(topics_data.values()), key=lambda x: x['name'])
            
            return JsonResponse({
                'topics': topics_list,
                'total_count': len(topics_list),
                'filters_applied': {
                    'grade': grade,
                    'search': search
                }
            })
            
        except Exception as e:
            logger.error(f"Error in GetTopicsView: {str(e)}")
            return JsonResponse({
                'error': 'Failed to load topics'
            }, status=500)


class PersonaListView(View):
    """Get available tutor personas"""
    
    def get(self, request):
        personas = TutorPersona.objects.all()
        return JsonResponse({
            'personas': [
                {
                    'name': persona.name,
                    'description': persona.description
                }
                for persona in personas
            ]
        })

@method_decorator(csrf_exempt, name='dispatch')
class RateAnswerView(View):
    """Rate a previous answer"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            qa_id = data.get('qa_id')
            rating = data.get('rating')
            
            qa = QuestionAnswer.objects.get(id=qa_id)
            qa.rating = rating
            qa.save()
            
            return JsonResponse({
                'message': 'Rating saved successfully',
                'qa_id': qa_id,
                'rating': rating
            })
            
        except QuestionAnswer.DoesNotExist:
            return JsonResponse({'error': 'Question-Answer not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

class QuestionHistoryView(View):
    """Get question history with pagination"""
    
    def get(self, request):
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 10))
        
        offset = (page - 1) * page_size
        
        questions = QuestionAnswer.objects.all().order_by('-created_at')[offset:offset + page_size]
        total_count = QuestionAnswer.objects.count()
        
        return JsonResponse({
            'questions': [
                {
                    'id': qa.id,
                    'question': qa.question,
                    'answer': qa.answer[:200] + '...' if len(qa.answer) > 200 else qa.answer,
                    'persona': qa.persona.name,
                    'rating': qa.rating,
                    'created_at': qa.created_at.isoformat(),
                    'sources_count': len(qa.retrieved_content)
                }
                for qa in questions
            ],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': (total_count + page_size - 1) // page_size
            }
        })
        
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