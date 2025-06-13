# documents/serializers.py
from rest_framework import serializers
from .models import Document, DocumentHistory
from django.utils.translation import gettext_lazy as _
from django.urls import reverse

class DocumentStatsSerializer(serializers.Serializer):
    word_count = serializers.IntegerField()
    character_count = serializers.IntegerField()
    page_count = serializers.IntegerField()
    reading_time = serializers.IntegerField()


class AIMarkerSerializer(serializers.Serializer):
    type = serializers.CharField()
    confidence = serializers.FloatField()
    sections = serializers.ListField(child=serializers.CharField())

class SourceMatchSerializer(serializers.Serializer):
    source = serializers.CharField()
    url = serializers.URLField()
    match_percentage = serializers.FloatField()
    snippets = serializers.ListField(child=serializers.CharField())

class TextAnalysisSerializer(serializers.Serializer):
    original_content = serializers.FloatField()
    plagiarized_content = serializers.FloatField()
    ai_generated_content = serializers.FloatField()
class DocumentSerializer(serializers.ModelSerializer):
    fileUrl = serializers.SerializerMethodField()
    highlights = serializers.SerializerMethodField()
    originality_score = serializers.SerializerMethodField()
    documentStats = DocumentStatsSerializer(source='*', read_only=True)
    
    # Expose content and document_code for read operations
    content = serializers.CharField(read_only=True)
    document_code = serializers.CharField(read_only=True) # Expose document_code

    class Meta:
        model = Document 
        fields = [
            'id', 'fileUrl', 'content', 'content_hash', 'plagiarism_score', 'ai_score', 
            'originality_score', 'documentStats', 'highlights', 'document_code', 
            'title', 'created_at', 'recipient_email'
        ]
        read_only_fields = ['user', 'plagiarism_score', 'ai_score', 'originality_score',
                            'highlights', 'documentStats', 'document_code', 'created_at']


    def get_fileUrl(self, obj):
        request = self.context.get('request')
        if obj.file:
            if request:
                return request.build_absolute_uri(obj.file.url)
            return obj.file.url
        return None

    def get_format(self, obj):
        if obj.file:
            return obj.file.name.split('.')[-1].lower()
        return 'unknown'

    def get_highlights(self, obj):
        return obj.highlights

    def get_originality_score(self, obj):
        return round(max(0.0, 100.0 - (obj.plagiarism_score + obj.ai_score)), 1)



class DocumentHistorySerializer(serializers.ModelSerializer):
    # Ensure DocumentSerializer used here includes necessary fields like 'document_code'
    document = DocumentSerializer(read_only=True) # Use updated DocumentSerializer
    document_name = serializers.SerializerMethodField()
    
    # Ensure stats are included in history records as well
    documentStats = DocumentStatsSerializer(source='*', read_only=True) 


    class Meta:
        model = DocumentHistory
        fields = [
            'id', 'document', 'created_at', 'plagiarism_score', 'ai_score',
            'originality_score', 'highlights', 'content', 'document_name',
            'documentStats' 
        ]
        read_only_fields = ['document', 'created_at', 'highlights', 'content'] 

    def get_document_name(self, obj):
        if obj.document and obj.document.title: 
            return obj.document.title
        elif obj.document and obj.document.file:
            return obj.document.file.name.split("/")[-1] 
        return None

    def get_originality_score(self, obj):
        return round(max(0.0, 100.0 - (obj.plagiarism_score + obj.ai_score)), 1)