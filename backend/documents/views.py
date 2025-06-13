# documents/viewsets.py
import hashlib
import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAuthenticatedOrReadOnly
from django.db import transaction
from django.core.mail import send_mail
from django.conf import settings 
from .models import Document, DocumentHistory
from .serializers import DocumentSerializer, DocumentHistorySerializer
from .utils import (
    extract_text_from_file,
    analyze_text_for_plagiarism_and_ai 
)


logger = logging.getLogger(__name__)

class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    permission_classes = [IsAuthenticatedOrReadOnly] 

    def get_queryset(self):
        if self.request.user.is_authenticated:
            return Document.objects.filter(user=self.request.user).order_by('-created_at')
        return Document.objects.none()

    @transaction.atomic 
    def create(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        recipient_email = request.data.get('recipient_email')
        send_report_to_other = request.data.get('send_report_to_other', 'false').lower() == 'true'

        if not file:
            return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

        # 1. File Size Check 
        if file.size > 10 * 1024 * 1024:
            return Response({"error": "File too large (max 10MB)"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 2. Extract Text and Calculate Hash
            file_content = extract_text_from_file(file).strip()
            content_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
            file_name = file.name
            user = request.user if request.user.is_authenticated else None

            # 3. Basic Content Validation 
            words = file_content.split() # Simple split for word count
            if len(words) < 10 or len(file_content) < 200:
                return Response({"error": "Document too short for analysis (min 10 words, 200 characters)."}, status=status.HTTP_400_BAD_REQUEST)

            # 4. Check for Existing Document (Deduplication)
            existing_document = None
            if user:
                # Prioritize existing document for the current user
                existing_document = Document.objects.filter(user=user, content_hash=content_hash).first()
            if not existing_document:
                # If no user-specific document, check for a public document (user=None)
                existing_document = Document.objects.filter(user__isnull=True, content_hash=content_hash).first()


            if existing_document:
                logger.info(f"Existing document found for hash {content_hash} (ID: {existing_document.id}). Updating/Adding history.")
                document = existing_document
                document.title = file_name
                document.file = file
                document.content = file_content 

            else:
                logger.info(f"Creating new document for hash {content_hash}.")
                document = Document(
                    user=user,
                    title=file_name,
                    file=file,
                    content=file_content,
                    content_hash=content_hash,
                    recipient_email=recipient_email if send_report_to_other else None 
                )
                document.save() 

            plagiarism_score, ai_score, originality_score, highlights, stats = \
                analyze_text_for_plagiarism_and_ai(file_content)

            document.plagiarism_score = plagiarism_score
            document.ai_score = ai_score
            document.originality_score = originality_score
            document.highlights = highlights 
            document.word_count = stats['word_count']
            document.character_count = stats['character_count']
            document.page_count = stats['page_count']
            document.reading_time = stats['reading_time']
            
            document.save()
            logger.info(f"Document {document.id} analysis saved: Plag={plagiarism_score}%, AI={ai_score}%, Originality={originality_score}%")

            DocumentHistory.objects.create(
                document=document,
                content=file_content, 
                plagiarism_score=plagiarism_score,
                ai_score=ai_score,
                originality_score=originality_score,
                highlights=highlights,
                word_count = stats['word_count'],
                character_count = stats['character_count'],
                page_count = stats['page_count'],
                reading_time = stats['reading_time'],
            )
            logger.info(f"Created DocumentHistory record for document {document.id}")

            # 8. Send Analysis Report Email
            if send_report_to_other and recipient_email:
                self._send_analysis_report_email(document.document_code, recipient_email)
            elif user and not send_report_to_other: 
                self._send_analysis_report_email(document.document_code, user.email)

            # 9. Return Response
            serializer = self.get_serializer(document)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except UnicodeDecodeError:
            return Response({"error": "Could not decode file content as UTF-8. Please upload a plain text or properly encoded document."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.exception("Error during document analysis or saving in DocumentViewSet.create") 
            return Response({"error": f"An unexpected error occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def my_documents(self, request):
        """
        Retrieves all documents uploaded by the authenticated user.
        """
        documents = self.get_queryset() 
        serializer = self.get_serializer(documents, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'])
    def check_document_history(self, request):
        """
        Retrieves document details and history for a given document_code.
        Can be accessed by anyone with the code.
        """
        document_code = request.query_params.get('document_code', None)
        if not document_code:
            return Response({"error": "Document code is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            document = Document.objects.prefetch_related('history_records').get(document_code=document_code)
            serializer = self.get_serializer(document)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Document.DoesNotExist:
            return Response({"error": "Document not found with the provided code."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error checking document history for {document_code}: {e}", exc_info=True)
            return Response({"error": "An error occurred while retrieving document history."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _send_analysis_report_email(self, document_code, recipient_email):
        subject = f"Plagiarism and AI Analysis Report for Document Code: {document_code}"
        report_url = f"{settings.FRONTEND_URL}/history?code={document_code}" # Assuming you set FRONTEND_URL in Django settings
        message = (f"Dear User,\n\n"
                f"Your document (Code: {document_code}) has been analyzed.\n"
                f"You can view the full report by clicking here: {report_url}\n\n"
                f"Thank you,\nYour Plagiarism Checker Team")
        
        # Ensure DEFAULT_FROM_EMAIL is set in your Django settings.py
        from_email = settings.DEFAULT_FROM_EMAIL 
        try:
            send_mail(subject, message, from_email, [recipient_email], fail_silently=False)
            logger.info(f"Analysis report sent to {recipient_email} for document {document_code}")
        except Exception as e:
            logger.error(f"Failed to send email report to {recipient_email} for document {document_code}: {e}")