# documents/models.py
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    plagiarism_score = models.FloatField()
    ai_score = models.FloatField()
    _highlights = models.JSONField(default=list)
    content_hash= models.CharField(max_length=64, unique=True)
    file = models.FileField(upload_to='documents/')
    created_at = models.DateTimeField(auto_now_add=True)
    word_count = models.IntegerField()
    character_count = models.IntegerField()
    page_count = models.IntegerField()
    reading_time = models.IntegerField()
    @property
    def highlights(self):
        return self._highlights

class DocumentHistory(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    content = models.TextField()
    plagiarism_score = models.FloatField()
    ai_score = models.FloatField()
    highlights = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
    def __str__(self):
        return f"DocumentHistory for {self.document.id} at {self.created_at}"
