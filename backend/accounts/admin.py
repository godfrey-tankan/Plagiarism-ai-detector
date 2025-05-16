from django.contrib import admin
from .models import CustomUser, UserActivity



# Register your models here.

@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'is_verified', 'is_admin', 'user_type')
    list_filter = ('is_verified', 'is_admin', 'user_type')
    search_fields = ('username', 'email')
    ordering = ('-date_joined',)
    fieldsets = (
        (None, {'fields': ('username', 'email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'avatar')}),
        ('Permissions', {'fields': ('is_verified', 'is_admin', 'is_active', 'is_staff', 'is_superuser')}),
        ('User Type', {'fields': ('user_type',)}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'password1', 'password2')}
         ),
    )
