import json
import sqlite3
import numpy as np
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
url = "http://localhost:8080/embed"
headers = {"Content-Type": "application/json"}

DATABASE = "embeddings.db"
EMBEDDING_DIM = 5888


def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_type TEXT,
            content TEXT,
            embedding TEXT
        )""")


init_db()


def generate_embedding(body):
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        data = response.json()
        embeddings = data.get("embeddings", [])
        if embeddings:
            embedding_array = np.array(embeddings, dtype=np.float32)
            if embedding_array.ndim > 1:
                embedding_array = embedding_array[0]
            return embedding_array
    print(f"Error from embedding service: {response.status_code}")
    return np.array([], dtype=np.float32)


def insert_embedding(content_type, content, embedding):
    embedding_json = json.dumps(embedding.tolist())
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            "INSERT INTO embeddings (content_type, content, embedding) VALUES (?, ?, ?)",
            (content_type, content, embedding_json),
        )


def get_all_embeddings():
    with sqlite3.connect(DATABASE) as conn:
        rows = conn.execute(
            "SELECT id, content_type, content, embedding FROM embeddings"
        ).fetchall()
    results = [
        {
            "id": r[0],
            "content_type": r[1],
            "content": r[2],
            "embedding": zero_pad_embedding(
                np.array(json.loads(r[3]), dtype=np.float32)
            ),
        }
        for r in rows
    ]
    return results


def zero_pad_embedding(embedding, desired_length=EMBEDDING_DIM):
    length = embedding.shape[0]
    if length > desired_length:
        return embedding[:desired_length]
    elif length < desired_length:
        return np.concatenate(
            [embedding, np.zeros(desired_length - length, dtype=np.float32)]
        )
    return embedding


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400

    embedding = generate_embedding(data)
    if embedding.size == 0:
        return jsonify({"error": "Could not generate embedding"}), 400

    content_type = "text" if "text" in data else "image" if "image" in data else None
    if not content_type:
        return jsonify({"error": "No 'text' or 'image' field provided"}), 400

    content = data[content_type]
    embedding = zero_pad_embedding(embedding)
    insert_embedding(content_type, content, embedding)

    return jsonify({"message": "Content embedded and stored successfully"}), 200


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400

    embed_body = (
        {"text": data["text"]}
        if "text" in data
        else {"image": data["image"]}
        if "image" in data
        else None
    )
    if not embed_body:
        return jsonify({"error": "No 'text' or 'image' field provided"}), 400

    query_embedding = generate_embedding(embed_body)
    query_embedding = zero_pad_embedding(query_embedding, desired_length=EMBEDDING_DIM)
    if query_embedding.size == 0:
        return jsonify({"error": "Could not generate query embedding"}), 400

    all_embs = get_all_embeddings()
    similarities = [
        (item, cosine_similarity(query_embedding, item["embedding"]))
        for item in all_embs
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_10 = similarities[:10]

    results = [
        {
            "id": obj["id"],
            "content_type": obj["content_type"],
            "content": obj["content"],
            "score": float(score),
        }
        for obj, score in top_10
    ]

    return jsonify({"results": results}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
