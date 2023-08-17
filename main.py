import nltk
import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def calculate_similarity(query, sentences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = np.dot(query_vector, tfidf_matrix.T).toarray()[0]
    return similarity_scores

def get_best_answer(query, content_list):
    sentences = []
    for content in content_list:
        sentences.extend(preprocess_text(content))
    similarity_scores = calculate_similarity(query, sentences)
    most_similar_index = np.argmax(similarity_scores)
    return sentences[most_similar_index]

# Function to scrape content from a webpage (Replace this with your web scraping logic)
def scrape_web_page(url):
    # Add your web scraping logic here
    # Return the extracted content as a string
    return "Content from " + url

def lambda_handler(event, context):
    print(event)
    user_question = event["question"]

    # Assuming you have fetched the related sites URLs
    related_sites = [
        "https://www.example.com/site1",
        "https://www.example.com/site2",
        # Add more URLs
    ]

    # Extract content from the related sites
    site_contents = []
    for url in related_sites:
        content = scrape_web_page(url)
        site_contents.append(content)

    # Assuming you have determined that the user's question is related to parrots
    if "parrot" in user_question.lower():
        best_answer = get_best_answer(user_question, site_contents)
        return {
            "statusCode": 200,
            "body": json.dumps({"best_answer": best_answer})
        }
    else:
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Not a parrot-related question."})
        }
