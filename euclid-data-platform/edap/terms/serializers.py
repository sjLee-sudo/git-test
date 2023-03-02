from django.db import models
from django.db.models import fields
from rest_framework import serializers
from .models import Terminology, Synonym, Representative, Stopword, Compound, EdgelistTerm

class RecursiveField(serializers.Serializer):
    def to_representation(self, value):
        serializer = self.parent.parent.__class__(value, context=self.context)
        return serializer.data

class TerminologySerializer(serializers.ModelSerializer):
    class Meta:
        model = Terminology
        fields = ('id','term',)

class SynonymSerializer(serializers.ModelSerializer):

    class Meta:
        model = Synonym
        fields = ('id','main_term','sub_term',)

class RepresentativeSerializer(serializers.ModelSerializer):

    class Meta:
        model = Representative
        fields = ('id','main_term','sub_term',)

class StopwordSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stopword
        fields = ('id','stopword',)

class CompoundSerializer(serializers.ModelSerializer):
    class Meta:
        model = Compound
        fields = ('id','compound','component')

class EdgelistTermSerializer(serializers.ModelSerializer):
    class Meta:
        model = EdgelistTerm
        fields = ('id','term','category')
