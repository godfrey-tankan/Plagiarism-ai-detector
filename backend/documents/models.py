# documents/models.py
from django.db import models
from django.contrib.auth import get_user_model
import uuid
import hashlib

User = get_user_model()

class Document(models.Model):
    document_code = models.CharField(max_length=12, unique=True, blank=True, null=True)
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='documents') 
    
    title = models.CharField(max_length=255, blank=True, null=True)
    file = models.FileField(upload_to='documents/')
    content = models.TextField()
    content_hash = models.CharField(max_length=64, unique=True) 

    # Analysis results
    plagiarism_score = models.FloatField(default=0.0) 
    ai_score = models.FloatField(default=0.0) 
    originality_score = models.FloatField(default=0.0) 
    highlights = models.JSONField(default=list)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    word_count = models.IntegerField(default=0) 
    character_count = models.IntegerField(default=0) 
    page_count = models.IntegerField(default=0) 
    reading_time = models.IntegerField(default=0) 

    recipient_email = models.EmailField(max_length=255, blank=True, null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title or f"Document {self.document_code or self.id}"

    def save(self, *args, **kwargs):
        if not self.document_code:
            self.document_code = str(uuid.uuid4()).replace('-', '')[:12].upper()
            while Document.objects.filter(document_code=self.document_code).exists():
                self.document_code = str(uuid.uuid4()).replace('-', '')[:12].upper()
        
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode('utf-8')).hexdigest()

        super().save(*args, **kwargs)

class DocumentHistory(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='history_records')
    content = models.TextField()
    plagiarism_score = models.FloatField(default=0.0) 
    ai_score = models.FloatField(default=0.0) 
    originality_score = models.FloatField(default=0.0)
    highlights = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    
    word_count = models.IntegerField(default=0)
    character_count = models.IntegerField(default=0)
    page_count = models.IntegerField(default=0)
    reading_time = models.IntegerField(default=0)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Document Histories"

    def __str__(self):
        return f"History for {self.document.title or self.document.document_code} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"