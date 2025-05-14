from rest_framework.views import APIView
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ValidationError

from .models import Document, DocumentHistory
from .serializers import DocumentSerializer, DocumentHistorySerializer
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated

from .utils import (
    extract_text_from_file,
    analyze_text,
    check_ai_probability,
    calculate_document_stats
)

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import hashlib
import re
import logging

logger = logging.getLogger(__name__)

def calculate_content_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


@method_decorator(csrf_exempt, name='dispatch')
class AnalyzeDocumentView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            # 1. file in request?
            if 'document' not in request.FILES:
                return Response({"error": "No document provided"}, status=400)

            file = request.FILES['document']
            # 2. size limit
            if file.size > 10 * 1024 * 1024:
                return Response({"error": "File too large (max 10MB)"}, status=400)

            # 3. extract & basic validation
            text = extract_text_from_file(file).strip()
            logger.info(f"[{request.user}] extracted {len(text)} chars")

            words = re.findall(r'\w+', text)
            if len(words) < 10 or len(text) < 200:
                return Response({"error": "Document too short for analysis"}, status=400)

            # 4. dedupe by hash
            content_hash = calculate_content_hash(text)
            existing = Document.objects.filter(content_hash=content_hash).first()

            # 5. plagiarism & AI
            plag = analyze_text(content_hash, text)
            p_score = min(plag['score'], 100.0)

            ai = check_ai_probability(text, plag['highlights'], plagiarism_score=p_score)
            ai_score = min(ai['score'], 100.0 - p_score)

            # 6. original
            orig = round(max(0.0, 100.0 - (p_score + ai_score)), 1)

            p_score = round(p_score, 1)
            ai_score = round(ai_score, 1)

            # 7. persist
            stats = calculate_document_stats(text)
            highlights = plag['highlights'] + ai['highlights']

            if existing:
                if existing.user != request.user and not request.user.is_superuser:
                    result = {
                        'id': existing.id,
                        'fileUrl': existing.file.url,
                        'plagiarismScore': 100.0,
                        'aiScore': existing.ai_score,
                        'originalScore': 0.0,
                        'documentStats': {
                            'wordCount': existing.word_count,
                            'characterCount': existing.character_count,
                            'pageCount': existing.page_count,
                            'readingTime': existing.reading_time
                        },
                        'highlights': existing.highlights
                    }
                    return Response(result, status=status.HTTP_200_OK)
                    return Response({"error": "Document already exists"}, status=400)
                existing.plagiarism_score = p_score
                existing.ai_score = ai_score
                existing._highlights = highlights
                existing.word_count = stats['word_count']
                existing.character_count = stats['character_count']
                existing.page_count = stats['page_count']
                existing.reading_time = stats['reading_time']
                existing.save()
                doc = existing
            else:
                doc = Document.objects.create(
                    user=request.user,
                    content=text,
                    content_hash=content_hash,
                    plagiarism_score=p_score,
                    ai_score=ai_score,
                    _highlights=highlights,
                    file=file,
                    **stats
                )
            try:
                DocumentHistory.objects.create(
                    document=doc,
                    content=doc.content,
                    plagiarism_score=doc.plagiarism_score,
                    ai_score=doc.ai_score,
                    highlights=doc.highlights 
                )
            except Exception as e:
                logger.error(f"Failed to create DocumentHistory: {e}")

            # 8. response (exact same shape you had)
            result = {
                'id': doc.id,
                'fileUrl': doc.file.url,
                'plagiarismScore': p_score,
                'aiScore': ai_score,
                'originalScore': orig,
                'documentStats': {
                    'wordCount': doc.word_count,
                    'characterCount': doc.character_count,
                    'pageCount': doc.page_count,
                    'readingTime': doc.reading_time
                },
                'highlights': doc.highlights
            }
            return Response(result, status=200)

        except ValidationError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.exception("AnalyzeDocumentView error")
            return Response({"error": "Internal server error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        file = self.request.FILES.get('file')
        if file:
            text = extract_text_from_file(file)
            content_hash = calculate_content_hash(text)

            existing = Document.objects.filter(
                user=self.request.user,
                content_hash=content_hash
            ).first()

            if existing:
                # update scores if re-uploaded
                plag = analyze_text(content_hash, text)
                ai = check_ai_probability(text, plag['highlights'], plagiarism_score=plag['score'])
                existing.plagiarism_score = plag['score']
                existing.ai_score = ai['score']
                existing._highlights = plag['highlights'] + ai['highlights']
                existing.save()
                return

            stats = calculate_document_stats(text)
            serializer.save(
                user=self.request.user,
                content=text,
                content_hash=content_hash,
                _highlights=[],
                **stats
            )

    @action(detail=False, methods=['get'], url_path='test-csrf')
    def test_csrf(self, request):
        return Response({"message": "CSRF exemption works!"}, status=status.HTTP_200_OK)


@method_decorator(csrf_exempt, name='dispatch')
class UserDocumentHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        histories = DocumentHistory.objects.filter(document__user=request.user).select_related('document')
        serializer = DocumentHistorySerializer(histories, many=True)
        return Response(serializer.data)