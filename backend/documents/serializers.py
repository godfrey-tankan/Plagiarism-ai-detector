# documents/serializers.py
from rest_framework import serializers
from .models import Document, DocumentHistory
from django.utils.translation import gettext_lazy as _
from django.urls import reverse

class DocumentStatsSerializer(serializers.Serializer):
    word_count = serializers.IntegerField(source='word_count')
    character_count = serializers.IntegerField(source='character_count')
    page_count = serializers.IntegerField(source='page_count')
    reading_time = serializers.IntegerField(source='reading_time')

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
    original_score = serializers.SerializerMethodField()
    documentStats = DocumentStatsSerializer(source='*', read_only=True)


    class Meta:
        model = Document
        fields = [
            'id', 'fileUrl', 'content', 'plagiarism_score', 'ai_score',
            'original_score', 'documentStats', 'highlights'
        ]
        read_only_fields = ['user', 'content_hash', 'content']


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

    def get_original_score(self, obj):
        return round(max(0.0, 100.0 - (obj.plagiarism_score + obj.ai_score)), 1)


class DocumentHistorySerializer(serializers.ModelSerializer):
    document = DocumentSerializer(read_only=True)
    document_name = serializers.SerializerMethodField()

    class Meta:
        model = DocumentHistory
        fields = ['id', 'document', 'created_at', 'plagiarism_score', 'ai_score', 'highlights', 'content', 'document_name']

    def get_document_name(self, obj):
        if obj.document and obj.document.file:
            return obj.document.file.name.split("/")[-1]
        return None