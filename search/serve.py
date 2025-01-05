import base64
import io
from itertools import chain
import json
import sqlite3
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np


app = Flask(__name__)
CORS(app)
url = "http://localhost:8080/embed"
headers = {"Content-Type": "application/json"}

DATABASE = 'embeddings.db'
EMBEDDING_DIM = 5888

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content_type TEXT,
        content TEXT,
        embedding TEXT
    )''')
    conn.commit()
    conn.close()

init_db()


def generate_embedding(body):
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        data = response.json()
        embeddings = data.get("embeddings", [])
        if len(embeddings) > 0:
            embedding_array = np.array(embeddings, dtype=np.float32)
            print(f"Generated embedding shape: {embedding_array.shape}")
            # If we get a 2D array with shape (1, N), take the first row
            if len(embedding_array.shape) > 1:
                embedding_array = embedding_array[0]
            print(f"Final embedding shape: {embedding_array.shape}")
            return embedding_array
        else:
            return np.array([], dtype=np.float32)
    else:
        print(f"Error from embedding service: {response.status_code}")
        return np.array([], dtype=np.float32)

def insert_embedding(content_type, content, embedding):
    print(f"Inserting embedding with shape: {embedding.shape}")
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    embedding_json = json.dumps(embedding.tolist())
    c.execute("INSERT INTO embeddings (content_type, content, embedding) VALUES (?, ?, ?)",
              (content_type, content, embedding_json))
    conn.commit()
    conn.close()

def get_all_embeddings():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT id, content_type, content, embedding FROM embeddings")
    rows = c.fetchall()
    conn.close()

    results = []
    for r in rows:
        emb_array = np.array(json.loads(r[3]), dtype=np.float32)
        # Ensure the embedding is zero-padded after retrieval
        emb_array = zero_pad_embedding(emb_array, desired_length=EMBEDDING_DIM)
        results.append({
            "id": r[0],
            "content_type": r[1],
            "content": r[2],
            "embedding": emb_array
        })
    return results



def zero_pad_embedding(embedding, desired_length=EMBEDDING_DIM):
    embedding = np.array(embedding, dtype=np.float32)
    length = embedding.shape[0]

    if length > desired_length:
        # Truncate
        return embedding[:desired_length]
    elif length < desired_length:
        # Zero-pad
        padding = np.zeros(desired_length - length, dtype=np.float32)
        return np.concatenate([embedding, padding])
    else:
        return embedding

def cosine_similarity(v1, v2):
    # v1 and v2 should now always match dimension
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Generate embedding
    embedding = generate_embedding(data)
    if embedding.size == 0:
        return jsonify({"error": "Could not generate embedding"}), 400

    if 'text' in data:
        content_type = 'text'
        content = data['text']
    elif 'image' in data:
        content_type = 'image'
        content = data['image']
    else:
        return jsonify({"error": "No 'text' or 'image' field provided"}), 400

    embedding = zero_pad_embedding(embedding)
    insert_embedding(content_type, content, embedding)

    return jsonify({"message": "Content embedded and stored successfully"}), 200

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400

    print("\n=== Starting new search ===")
    print(f"Search request data: {data}")

    if 'text' in data:
        embed_body = {"text": data['text']}
    elif 'image' in data:
        embed_body = {"image": data['image']}
    else:
        return jsonify({"error": "No 'text' or 'image' field provided"}), 400

    query_embedding = generate_embedding(embed_body)
    query_embedding = zero_pad_embedding(query_embedding, desired_length=EMBEDDING_DIM)
    print(f"Query embedding shape: {query_embedding.shape}")

    if query_embedding.size == 0:
        return jsonify({"error": "Could not generate query embedding"}), 400

    all_embs = get_all_embeddings()
    similarities = []

    for item in all_embs:
        try:
            sim = cosine_similarity(query_embedding, item["embedding"])
            similarities.append((item, sim))
        except Exception as e:
            print(f"Error comparing with item {item['id']}: {str(e)}")
            continue

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_10 = similarities[:10]

    results = []
    for obj, score in top_10:
        results.append({
            "id": obj["id"],
            "content_type": obj["content_type"],
            "content": obj["content"],
            "score": float(score)
        })

    return jsonify({"results": results}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
