from django.db import models
from documents.models import Document

class PlayerHierarchy(models.Model):
    superior_player = models.ForeignKey('Player', models.CASCADE, related_name='superior_player_set', help_text='상위플레이어')
    sub_player = models.ForeignKey('Player', models.CASCADE, related_name='sub_player_set', help_text='소속플레이어')
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = 'player_hierarchy'
        unique_together = ('superior_player','sub_player')


class PlayerDocument(models.Model):
    player = models.ForeignKey('Player', models.CASCADE, related_name='player_set', help_text='플레이어아이디')
    document = models.ForeignKey(Document, models.CASCADE, db_column='doc_id', related_name='player_document_set', help_text='문서아이디')
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = 'player_document'
        unique_together = ('player','document')

class Player(models.Model):
    """
    """
    player_code = models.CharField(max_length=500, help_text='')
    player_name = models.CharField(max_length=500, blank=True, null=True, help_text='')
    documents = models.ManyToManyField(Document, through='PlayerDocument', through_fields=('player','document'), related_name='player_related_document')
    player_hierarchy = models.ManyToManyField('self', through='PlayerHierarchy', through_fields=('superior_player_id','sub_player_id'), symmetrical=False, related_name='player_relationship')
    remark = models.CharField(max_length=5000, blank=True, null=True, help_text='')
    updated_at = models.DateTimeField(auto_now=True, blank=True, null=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True, blank=True, null=True, editable=False)

    class Meta:
        db_table = 'player'
        indexes = [
            models.Index(fields=['player_code']),
            models.Index(fields=['player_name']),
        ]