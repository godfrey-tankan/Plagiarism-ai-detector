# Generated by Django 5.2 on 2025-05-14 14:47

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0004_document__highlights'),
    ]

    operations = [
        migrations.CreateModel(
            name='DocumentHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
                ('plagiarism_score', models.FloatField()),
                ('ai_score', models.FloatField()),
                ('highlights', models.JSONField(default=list)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('document', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='documents.document')),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
    ]
