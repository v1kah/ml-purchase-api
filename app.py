from flask import Flask, request, jsonify
from gensim.models import Word2Vec

# Завантаження моделі
model = Word2Vec.load("universal_word2vecFinal.model")

app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    token = request.args.get("token")
    if not token:
        return jsonify({"error": "token parameter is required"}), 400
    
    try:
        similar = model.wv.most_similar(token, topn=5)
        recommendations = [item for item, score in similar]
        return jsonify({
            "input": token,
            "recommendations": recommendations
        })
    except KeyError:
        return jsonify({"error": f"Token '{token}' not found in model."}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
