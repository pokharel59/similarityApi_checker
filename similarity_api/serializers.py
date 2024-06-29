from rest_framework import serializers
from .models import UserText, SimilarityScore

class UserTextSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserText
        fields = ['id', 'name', 'text']

class SimilarityScoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimilarityScore
        fields = ['id', 'user_text', 'compared_with', 'score']
