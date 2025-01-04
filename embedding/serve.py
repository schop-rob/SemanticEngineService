import base64
import io
import torch
from PIL import Image
from flask import Flask, request, jsonify
from colpali_engine.models import ColQwen2, ColQwen2Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "vidore/colqwen2-v0.1"
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()



processor = ColQwen2Processor.from_pretrained(LOCAL_PROCESSOR_PATH)

app = Flask(__name__)

@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json(force=True)

    def process_text(text):
        batch_queries = processor.process_queries([text]).to(model.device)
        with torch.no_grad():
            text_embeddings = model(**batch_queries)
        return text_embeddings.cpu().float().numpy().reshape(-1).tolist()

    def process_image(image_base64):
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        batch_images = processor.process_images([img]).to(model.device)
        with torch.no_grad():
            image_embeddings = model(**batch_images)
        return image_embeddings.cpu().float().numpy().reshape(-1).tolist()

    if "text" in data:
        embeddings = process_text(data["text"])
    elif "image" in data:
        embeddings = process_image(data["image"])
    else:
        return jsonify({"error": "No 'text' or 'image' field provided"}), 400

    return jsonify({"embeddings": embeddings})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)