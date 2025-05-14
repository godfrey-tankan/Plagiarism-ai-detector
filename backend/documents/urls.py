# documents/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AnalyzeDocumentView, DocumentViewSet, UserDocumentHistoryView
from django.conf import settings
from django.conf.urls.static import static



router = DefaultRouter()
router.register(r'documents', DocumentViewSet, basename='document')

urlpatterns = [
    path('', include(router.urls)),
    
    path('analyze/', AnalyzeDocumentView.as_view(), name='analyze-document'),
    path('history/', UserDocumentHistoryView.as_view(), name='document-history'),
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)