from django.http import JsonResponse
from .models import SimilarityScore, UserText
from .serializers import SimilarityScoreSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def calculate_similarity(new_text, previous_texts):
    # Preprocess new text
    preprocessed_new_text = preprocess_text(new_text)

    # Preprocess previous texts
    preprocessed_previous_texts = [preprocess_text(entry) for entry in previous_texts]

    # Add new text to previous texts
    preprocessed_all_texts = preprocessed_previous_texts + [preprocessed_new_text]

    # Vectorize texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_all_texts)

    # Calculate cosine similarity between new text and previous texts
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Get similarity scores as a list
    similarity_scores_list = similarity_scores[0].tolist()

    return similarity_scores_list

@api_view(['POST'])
def add_and_calculate_similarity(request):
    if request.method == "POST":
        user_name = request.data.get("name")
        user_response = request.data.get("text")
        prev_texts_data = request.data.get("prev_text")

        if not user_name or not user_response or not prev_texts_data:
            return Response({"error": "Name, text, and prev_text are required"}, status=status.HTTP_400_BAD_REQUEST)

        new_user_text = UserText(name=user_name, text=user_response)
        new_user_text.save()

        previous_texts = []
        for prev_text_data in prev_texts_data:
            prev_text = UserText(name=prev_text_data["name"], text=prev_text_data["text"])
            prev_text.save()
            previous_texts.append(prev_text)

        similarity_scores = calculate_similarity(new_user_text.text, [prev_text.text for prev_text in previous_texts])

        response_data = [
            {"name": prev_text.name, "text": prev_text.text, "similarity": int(score * 100)}
            for prev_text, score in zip(previous_texts, similarity_scores)
        ]

        for prev_text, score in zip(previous_texts, similarity_scores):
            SimilarityScore.objects.create(
                user_text=new_user_text,
                compared_with=prev_text,
                score=score
            )

        return Response(response_data, status=status.HTTP_201_CREATED)
    else:
        return Response({"error": "Invalid HTTP method"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)