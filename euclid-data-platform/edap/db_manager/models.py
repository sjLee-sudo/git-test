from django.db import models

class FrontReference(models.Model):
    data_key = models.CharField(max_length=1000, help_text='데이터 키')
    data_value = models.CharField(max_length=1000, blank=True, null=True, help_text='데이터 값')
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'front_db_reference'
