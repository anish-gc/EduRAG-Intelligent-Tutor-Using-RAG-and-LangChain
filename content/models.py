from django.db import models
from pgvector.django import VectorField
# Create your models here.

class Content(models.Model):
    title = models.CharField(max_length=255)
    topic = models.CharField(max_length=100)
    grade = models.CharField(max_length=10)
    content_text = models.TextField()
    file_path = models.CharField(max_length=500, null=True, blank=True)
    embedding = VectorField(dimensions=1536, null=True, blank=True)  # OpenAI embeddings
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'content'
        indexes = [
            models.Index(fields=['topic', 'grade']),
        ]
        
        
class QueryLog(models.Model):
    question = models.TextField()
    answer = models.TextField()
    rating = models.SmallIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)        