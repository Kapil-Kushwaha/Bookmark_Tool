from flask import Flask, render_template, request, redirect, url_for, jsonify
import csv
from math import ceil
import ast
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import json
import numpy as np
from typing import List
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from urllib.parse import quote


# Flask application initialization
app = Flask(__name__)

from flask_sqlalchemy import SQLAlchemy

app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(100), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Configuration file and default settings
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    'service': 'google',
    'google_api_key': '',
    'ollama_api_key': '',
    'ollama_base_url': 'http://localhost:11434',
    'ollama_llm_model': 'llama3.1',
    'ollama_embedding_model': 'all-minilm'
}
def load_config():
    """Load configuration from file or return default if file not found."""

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

# Check if bookmarks.csv exists, if not create one with necessary columns
if not os.path.exists('bookmarks.csv'):
    with open('bookmarks.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['link', 'summary', 'embedding', 'base_url', 'timestamp', 'tags'])

# Load configuration
config = load_config()
    
def get_embedding(text, service='jina'):
    if not text.strip():
        print("Skipped empty text for embedding.")
        return []

    if service == 'jina':
        try:
            url = "https://r.jina.ai/v1/embeddings"
            headers = {"Content-Type": "application/json"}
            payload = {
                "input": [text],
                "model": "jina-embedding-b-en-v1"
            }
            print("Sending to Jina:", text[:150])
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            return embedding
        except Exception as e:
            print("Error while embedding with jina:", e)
            return []


        
def embed_all_links(service="google"):
    """Update embeddings for all bookmarks if necessary."""
    sample_query_embedding = get_embedding("Test", service=service)
    bookmarks = read_bookmarks()
    if bookmarks and sample_query_embedding.shape != bookmarks[0]['embedding'].shape:
        for bookmark in bookmarks:
            embedding_input = f"{bookmark['link']} {bookmark['summary']} {' '.join(bookmark.get('tags', []))}"
            embedding = get_embedding(embedding_input, service=service)
            bookmark['embedding'] = embedding
        write_bookmarks(bookmarks)

def get_summary(url):
    try:
        api_url = f"https://r.jina.ai/{quote(url, safe='')}"
        print(f"Jina Summary API called: {api_url}")
        response = requests.get(api_url)
        response.raise_for_status()
        return response.text.strip()  # returns full HTML summary string
    except Exception as e:
        print(f"Error while summarizing with Jina AI: {e}")
        return "Summary not available."

def read_bookmarks():
    """Read bookmarks from CSV file."""
    bookmarks = []
    with open("bookmarks.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 6:
                bookmarks.append({
                    'link': row[0],
                    'summary': row[1],
                    'embedding': np.array(ast.literal_eval(row[2])),
                    'base_url': row[3],
                    'timestamp': row[4],
                    'tags': ast.literal_eval(row[5]) if row[5] else []
                })
    return sorted(bookmarks, key=lambda x: x['timestamp'], reverse=True)

def write_bookmarks(bookmarks):
    """Write bookmarks to CSV file."""
    with open("bookmarks.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['link', 'summary', 'embedding', 'base_url', 'timestamp', 'tags'])
        for bookmark in bookmarks:
            writer.writerow([
                bookmark['link'],
                bookmark['summary'],
                bookmark['embedding'].tolist(),
                bookmark['base_url'],
                bookmark['timestamp'],
                bookmark['tags']
            ])
        
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
    
def semantic_search(query_vector: np.ndarray, document_vectors: List[np.ndarray]):
    """Perform semantic search using cosine similarity."""
    similarities = [
        (i, cosine_similarity(query_vector, doc_vector))
        for i, doc_vector in enumerate(document_vectors)
    ]
    return sorted(similarities, key=lambda x: x[1], reverse=True)

def add_new_bookmark(bookmark_url, existing_bookmarks):
    base_url = re.search(r'https?://([^/]+)', bookmark_url)
    base_url = base_url.group(1) if base_url else bookmark_url

    if config['service'] == 'jina':
        summary = get_summary(bookmark_url)
        tags = []  # Optional: you can leave tags empty or implement logic later

        embedding_bookmark = np.array(get_embedding(f"{bookmark_url} {summary}", service='jina'))
    else:
        summary, tags = get_summary(bookmark_url, service=config['service'], base_url=config.get('ollama_base_url', ''))
        embedding_bookmark = np.array(get_embedding(f"{bookmark_url} {summary} {' '.join(tags)}", service=config['service'], base_url=config.get('ollama_base_url', '')))
    
    new_bookmark = {
        'link': bookmark_url,
        'summary': summary,
        'embedding': embedding_bookmark,
        'base_url': base_url,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tags': tags
    }
    existing_bookmarks.append(new_bookmark)
    write_bookmarks(existing_bookmarks)
    return new_bookmark


@app.route('/')
@login_required
def index():
    """Render index page with paginated bookmarks."""
    bookmarks = read_bookmarks()
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total_pages = ceil(len(bookmarks) / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_bookmarks = bookmarks[start:end]
    return render_template('index.html', bookmarks=paginated_bookmarks, page=page, total_pages=total_pages)

@app.route('/remove/<path:link>')
@login_required
def remove_bookmark(link):
    """Remove a bookmark and redirect to index."""
    bookmarks = read_bookmarks()
    bookmarks = [b for b in bookmarks if b['link'] != link]
    write_bookmarks(bookmarks)
    return redirect(url_for('index'))

@app.route('/update_bookmark', methods=['POST'])
@login_required
def update_bookmark():
    """Update a bookmark's information."""
    data = request.json
    bookmarks = read_bookmarks()
    for bookmark in bookmarks:
        if bookmark['link'] == data['original_link']:
            bookmark['link'] = data['new_link']
            bookmark['summary'] = data['new_summary']
            write_bookmarks(bookmarks)
            return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/search')
@login_required
def search():
    """Perform semantic search on bookmarks."""
    query = request.args.get('query', '')
    bookmarks = read_bookmarks()
    query_embedding = get_embedding(query, service=config['service'], base_url=config['ollama_base_url'])
    corpus_embeddings = [b['embedding'] for b in bookmarks]
    hits = semantic_search(query_embedding, corpus_embeddings)
    sorted_bookmarks = [
        (bookmarks[idx], score) for idx, score in hits
    ]
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total_pages = ceil(len(sorted_bookmarks) / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_bookmarks = sorted_bookmarks[start:end]
    return render_template('search_results.html', 
                           bookmarks=paginated_bookmarks, 
                           query=query,
                           page=page,
                           total_pages=total_pages)

@app.route('/add_bookmark', methods=['GET', 'POST'])
@login_required
def add_bookmark():
    """Add new bookmarks."""
    if request.method == 'POST':
        bookmarks_text = request.form['bookmarks']
        bookmarks_list = bookmarks_text.split('\n')
        existing_bookmarks = read_bookmarks()
        for bookmark_url in bookmarks_list:
            bookmark_url = bookmark_url.strip()
            if bookmark_url and not any(b['link'] == bookmark_url for b in existing_bookmarks):
                add_new_bookmark(bookmark_url, existing_bookmarks)
        return redirect(url_for('index'))
    return render_template('add_bookmark.html')


@app.route('/config', methods=['GET', 'POST'])
def config_page():
    """Render and handle configuration page."""
    global config
    if request.method == 'POST':
        config['service'] = request.form['service']
        if config['service'] == 'google':
            config['google_api_key'] = request.form['google_api_key']
        else:
            config['ollama_base_url'] = request.form['ollama_base_url']
            config['ollama_llm_model'] = request.form['ollama_llm_model']
            config['ollama_embedding_model'] = request.form['ollama_embedding_model']
            config['ollama_api_key'] = request.form['ollama_api_key']
        save_config(config)
    return render_template('config.html', config=config)

@app.route('/embed_all_links_route', methods=['GET'])
def embed_all_links_route():
    """Endpoint to trigger embedding of all links."""
    embed_all_links(service=config['service'])
    return jsonify({"message": "All links have been re-embedded successfully."}), 200

@app.route('/test_models', methods=['GET'])
@login_required
def test_models():
    """Test embedding and summary generation."""
    try:
        test_text = "This is a test sentence for embedding."
        embedding = get_embedding(test_text, service=config['service'], base_url=config['ollama_base_url'])
        if not isinstance(embedding, np.ndarray) or embedding.size == 0:
            return jsonify({"message": "Embedding test failed. Check your configuration and try again."}), 400
        test_url = "https://example.com"
        summary, tags = get_summary(test_url, service=config['service'], base_url=config['ollama_base_url'])
        if not summary:
            return jsonify({"message": "Summary test failed. Check your configuration and try again."}), 400
        return jsonify({"message": "Models test successful. Embedding and summary generation are working correctly."}), 200
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/api/search', methods=['POST'])
def api_search():
    """
    Perform a semantic search on bookmarks.

    Request format:
    {
        "query": "search query string",
        "page": 1,  // optional, default is 1
        "per_page": 10  // optional, default is 10
    }

    Returns:
    {
        "results": [
            {
                "link": "bookmark url",
                "summary": "bookmark summary",
                "base_url": "base url of the bookmark",
                "timestamp": "bookmark creation timestamp",
                "tags": ["tag1", "tag2", ...],
                "similarity": float  // similarity score
            },
            ...
        ],
        "total_results": int,
        "page": int,
        "per_page": int,
        "total_pages": int
    }
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Missing query parameter"}), 400

        query = data['query']
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)

        bookmarks = read_bookmarks()
        query_embedding = np.array(get_embedding(query, service=config['service'], base_url=config['ollama_base_url']))
        
        corpus_embeddings = [b['embedding'] for b in bookmarks]
        hits = semantic_search(query_embedding, corpus_embeddings)
        results = [
            {**bookmarks[idx], 'similarity': float(score)}
            for idx, score in hits
        ]
        
        total_results = len(results)
        total_pages = ceil(total_results / per_page)
        
        start = (page - 1) * per_page
        end = start + per_page
        paginated_results = results[start:end]

        return jsonify({
            "results": [
                {
                    "link": result['link'],
                    "summary": result['summary'],
                    "base_url": result['base_url'],
                    "timestamp": result['timestamp'],
                    "tags": result.get('tags', []),
                    "similarity": result['similarity']
                } for result in paginated_results
            ],
            "total_results": total_results,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add', methods=['POST'])
def api_add_bookmark():
    """
    Add a new bookmark.

    Request format:
    {
        "url": "https://example.com"
    }

    Returns:
    - 201: Bookmark added successfully
    - 400: Missing url parameter or bookmark already exists
    - 500: Internal server error
    """
    try:
        data = request.json
        if not data or 'url' not in data:
            return jsonify({"error": "Missing url parameter"}), 400

        bookmark_url = data['url'].strip()
        existing_bookmarks = read_bookmarks()

        if any(b['link'] == bookmark_url for b in existing_bookmarks):
            return jsonify({"error": "Bookmark already exists"}), 400

        new_bookmark = add_new_bookmark(bookmark_url, existing_bookmarks)

        return jsonify({
            "message": "Bookmark added successfully",
            "bookmark": {
                "link": new_bookmark['link'],
                "summary": new_bookmark['summary'],
                "base_url": new_bookmark['base_url'],
                "timestamp": new_bookmark['timestamp'],
                "tags": new_bookmark['tags']
            }
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            return "Email already exists"
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        return "Invalid credentials"
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


with app.app_context():
    db.create_all()

with app.app_context():
    db.create_all()
    # Clear all sessions on app restart
    from flask_login import logout_user
    @app.before_request
    def clear_session_on_restart():
        if not current_user.is_authenticated:
            logout_user()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)