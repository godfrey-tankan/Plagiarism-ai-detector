from django.contrib import admin
from .models import Document,DocumentHistory

# Register your models here.

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('id','user', 'content_hash')
    search_fields = ('user','content_hash')
    list_filter = ('created_at',)
    ordering = ('-created_at',)
    date_hierarchy = 'created_at'

@admin.register(DocumentHistory)
class DocumentHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'document', 'created_at')
    search_fields = ('document__user', 'document__content_hash')
    list_filter = ('created_at',)
    ordering = ('-created_at',)
    date_hierarchy = 'created_at'