from flask import Flask, request, jsonify
import logging
import nltk
import json
import requests
import bs4
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
app = Flask(__name__)


# Configure logging
logging.basicConfig(level=logging.INFO)

tfidf_vectorizer = TfidfVectorizer(stop_words="english")

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

@app.route('/answer', methods=['POST'])
def get_answer():
    global sessions
    try:
        par_str = [
            "Macaw", "Cockatoo", "Budgerigar", "Budgie", "Conure",
            "Amazon", "Parrot", "Lorikeet", "Eclectus", "Kea", "Lovebird", "Parakeet",
            "Pionus", "Quaker", "Caique", "Ringneck", "Lory", "Rosella", "Cockatiel",
            "Ararauna", "Poicephalus"
        ]
        logging.info("39")
        data = request.get_json()
        ####
        # Check if the data contains a 'session_id' key
        remote_ip = request.remote_addr
        user_agent = request.user_agent.string
        if any(session.get("remote_ip") == remote_ip and session.get("user_agent") == user_agent for session in sessions):
            logging.info("40")
            # If 'session_id' is present,
            matching_session = next(
            (session for session in sessions if session.get("remote_ip") == remote_ip and session.get("user_agent") == user_agent),None)
            session_id = matching_session.get("session_id")
            #question = sessions.get("question")
            #answer = sessions.get("answer")
            #if session_id is None or question is None or answer is None:
            #    return jsonify({"message": "Missing session_id, question, or answer"}), 400
            #if session_id <= len(sessions):

            #else:
            #    logging.error("code 404 - message: Invalid session ID")
        else:
            # If 'session_id' is not present, treat it as a create_session request
            logging.info("56")
            session_id = len(sessions) + 1
            remote_ip = request.remote_addr
            user_agent = request.user_agent.string
            sessions.append({"session_id": session_id,"remote_ip":remote_ip ,"user_agent":user_agent, "qa_pairs": []})
            logging.info(f"code 201 - message: Session created, session_id: {session_id}")
        ####
        user_question = data["question"].lower()

        # Check if the user's question is related to parrots
        matching_parrots = [bird for bird in par_str if bird.lower() in user_question]
        if not matching_parrots:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Not a parrot-related question."})
            }

        if "question" in [session.get("question") for session in sessions]:
            # Collect all user questions and answers from the sessions
            user_questions = [session["question"] for session in sessions]
            user_answers = [session["answer"] for session in sessions]
            # Combine user questions and answers into a single search query
            search_query = " ".join(user_questions + user_answers ) 
            user_question = search_query + " " + user_question
        
        # Make a request to the external API for related URLs
        response = requests.post('https://0ai42tfv4e.execute-api.us-west-1.amazonaws.com/default/MyNLPSearchService', json={"subject":  user_question})
        response.json()
        response_data = response.json()
        body_list = json.loads(response_data['body'])
        related_sites = body_list

        # Extract content from the related sites
        site_contents = []
        for url in related_sites:
            content = scrape_web_page(url)
            site_contents.append(content)

        # Calculate result based on content availability
        cnt = sum(1 for content in site_contents if content)
        result = cnt / len(related_sites)

        # Calculate best answer based on question and content
        best_answer = get_best_answer(user_question, site_contents)

        # Append the user question and received answer to the sessions array
        sessions[session_id - 1]["qa_pairs"].append({"question": user_question, "answer": best_answer})
        logging.info(f"code 200 - message: QA pair added to session, session_id: {session_id}")
        return {
            "statusCode": 200,
            "body": json.dumps({
                "best_answer": best_answer,
                "result": result
            })
        }
    except Exception as error:
        logging.error("An error occurred:", error)
        return {
            "statusCode": 600,
            "body": json.dumps(f"An error occurred: {error}")
        }

def calculate_similarity(query, sentences):
    # Fit and transform the vectorizer with the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    # Transform the query using the same vectorizer
    query_vector = tfidf_vectorizer.transform([query])
    # Calculate similarity scores
    similarity_scores = np.dot(query_vector, tfidf_matrix.T).toarray()[0]
    return similarity_scores

def get_best_answer(query, content_list):
    try:
        sentences = [sentence for content in content_list for sentence in preprocess_text(content)]
        similarity_scores = calculate_similarity(query, sentences)
        most_similar_index = np.argmax(similarity_scores)
        return sentences[most_similar_index]
    except Exception as error:
                logging.error("An error occurred:", error)
                return {
                    "statusCode": 600,
                    "body": json.dumps(f"An error occurred:, {error}")
                    }


def scrape_web_page(url):
    try:
        response = requests.get(url, timeout=10)  # Set timeout in seconds
        if response.status_code == 200:
            soup = bs4.BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            logging.info("Scrape content from " + url)
            return content
            #return response.text  # Return the HTML content of the page
        else:
            logging.error("Failed to scrape content from " + url)
            return ''
    except requests.Timeout:
        logging.error("Timeout while scraping " + url)
        return ''
    except Exception as error:
        logging.error("An error occurred while scraping " + url + ": " + str(error))
        return ''
####
sessions = []
@app.route('/create_session', methods=['POST'])
def create_session():
    global sessions
    session_id = len(sessions) + 1
    sessions.append({"session_id": session_id, "qa_pairs": []})
    logging.info(f"message: Session created,session_id:{session_id}")
    response = {
        "message": "Session created",
        "session_id": session_id
    }
    return jsonify(response), 201

@app.route('/add_qa', methods=['POST'])
def add_qa():
    global sessions
    data = request.get_json()
    session_id = data.get("session_id")
    question = data.get("question")
    answer = data.get("answer")

    if session_id is None or question is None or answer is None:
        return jsonify({"message": "Missing session_id, question, or answer"}), 400

    if session_id <= len(sessions):
        sessions[session_id - 1]["qa_pairs"].append({"question": question, "answer": answer})
        return jsonify({"message": "QA pair added to session", "session_id": session_id}), 200
    else:
        return jsonify({"message": "Invalid session ID"}), 404
####
#def lambda_handler(event, context):AWS Lambda only

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    logging.info(f"Starting Flask app on {host}:{port}...")
    app.run(host=host, port=port)
