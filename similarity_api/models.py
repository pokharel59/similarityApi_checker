from django.db import models

class UserText(models.Model):
    name = models.CharField(max_length=100)
    text = models.TextField()

class SimilarityScore(models.Model):
    user_text = models.ForeignKey(UserText, on_delete=models.CASCADE, related_name='similarity_scores')
    compared_with = models.ForeignKey(UserText, on_delete=models.CASCADE, related_name='compared_similarity_scores')
    score = models.FloatField()
