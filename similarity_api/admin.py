from django.contrib import admin # type: ignore
from .models import SimilarityScore, UserText

admin.site.register(SimilarityScore)
admin.site.register(UserText)