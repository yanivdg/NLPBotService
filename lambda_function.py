import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_similarity(query, sentences):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    # Fit and transform the vectorizer with the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    # Transform the query using the same vectorizer
    query_vector = tfidf_vectorizer.transform([query])
    # Calculate similarity scores
    similarity_scores = np.dot(query_vector, tfidf_matrix.T).toarray()[0]
    return similarity_scores

def lambda_handler(event, context):
    similarity_scores = calculate_similarity(event['query'],event['sentences'])
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps(similarity_scores)
    }
