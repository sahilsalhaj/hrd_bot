from flask import Flask, render_template, request, jsonify
from api.hrd_bot_v1 import rag_pipeline, qa

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("question", "")

    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    print(f"Received query: {user_query}")  # Debugging output
    try:
        response = rag_pipeline(user_query, qa)
        print(f"Response: {response}")  # Debugging output
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in rag_pipeline: {e}")  # Debugging output
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
