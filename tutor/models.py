from django.db import models

# Create your models here.

class TutorPersona(models.Model):
    PERSONA_CHOICES = [
        ('friendly', 'Friendly'),
        ('strict', 'Strict'),
        ('humorous', 'Humorous'),
        ('encouraging', 'Encouraging'),
    ]
    
    name = models.CharField(max_length=50, choices=PERSONA_CHOICES, unique=True)
    system_prompt = models.TextField()
    description = models.TextField(blank=True)
    
    class Meta:
        db_table = 'tutor_personas'
        
        
        
        
class QuestionAnswer(models.Model):
    question = models.TextField()
    answer = models.TextField()
    persona = models.ForeignKey(TutorPersona, on_delete=models.CASCADE)
    retrieved_content = models.JSONField(default=list)
    rating = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'question_answers'        