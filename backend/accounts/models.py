from django.db import models

# Create your models here.
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    user_types = (
        ('admin', 'Admin'),
        ('student', 'Student'),
        ('supervisor', 'supervisor'),
    )
    is_verified = models.BooleanField(default=False)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    institution = models.CharField(max_length=100, blank=True)
    is_admin = models.BooleanField(default=False)
    user_type = models.CharField(max_length=20, choices=user_types, default='student')
    
    def __str__(self):
        return self.email
class UserActivity(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    last_login = models.DateTimeField(auto_now=True)
    last_activity = models.DateTimeField(auto_now=True)
    activity = models.TextField()
    class Meta:
        ordering = ['-last_activity']

    def __str__(self):
        return f"{self.user.username} - {self.last_activity}"