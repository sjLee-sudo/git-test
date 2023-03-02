from django.db import models

class Stopword(models.Model):
    stopword = models.CharField(max_length=100, default='', help_text='불용어')
    remark = models.CharField(max_length=1000, blank=True, null=True, help_text='비고')
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'stopword'
        indexes = [
            models.Index(fields=['stopword'])
        ]


class Representative(models.Model):
    main_term = models.CharField(max_length=100, help_text='대표어')
    sub_term = models.CharField(max_length=100, help_text='대표어 변환 대상 단어')
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'representative'
        indexes = [
            models.Index(fields=['main_term','sub_term'])
        ]
        unique_together = ('main_term','sub_term')

class Compound(models.Model):
    compound = models.CharField(max_length=100, default='', help_text='복합어')
    component = models.CharField(max_length=300, blank=True, null=True, help_text='복합어 구성 단어')
    remark = models.CharField(max_length=1000, blank=True, null=True, help_text='비고')
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'compound'
        indexes = [
            models.Index(fields=['compound'])
        ]
        unique_together = ('compound','component')


class Synonym(models.Model):
    main_term = models.CharField(max_length=100, help_text='동의어1')
    sub_term = models.CharField(max_length=100, help_text='동의어2')
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'synonym'
        indexes = [
            models.Index(fields=['main_term','sub_term'])
        ]        
        unique_together = ('main_term','sub_term')

class Terminology(models.Model):
    term = models.CharField(max_length=100, help_text='단어')
    remark = models.CharField(max_length=1000, blank=True, null=True, help_text='비고')
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'terminology'
        indexes = [
                models.Index(fields=['term'])
        ]        


class EdgelistTerm(models.Model):
    term = models.CharField(max_length=100, help_text='단어')
    category = models.CharField(max_length=100, blank=True, null=True, help_text='카테고리')
    remark = models.CharField(max_length=1000, blank=True, null=True, help_text='비고')
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'edgelist_term'
        indexes = [
                models.Index(fields=['term']),
                models.Index(fields=['category']),
                models.Index(fields=['category','term']),
        ]
        