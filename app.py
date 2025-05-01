from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from whitenoise import WhiteNoise
import pandas as pd
import openai
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/", prefix="/", index_file="index.html", autorefresh=True)

API_KEY = os.getenv("API_KEY")
print(API_KEY)

# @app.route('/', methods=['GET'])
# def hello():
#   return make_response("Hello, world!!!!!!!!!!!!!!!!!!!!")

compat = pd.read_csv("static/mbti_compatibility_matrix.csv", index_col=0)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")



@app.route("/recommend-mbti", methods=["POST"])
def recommend_mbti():
    data = request.get_json()
    team = data.get("team", [])

    if not team:
        return jsonify({"error": "Team is empty"}), 400

    prompt = (
        f"Given the current MBTI team: {', '.join(team)}, "
        "suggest one MBTI type that would best complement and improve this team's balance. "
        "Consider interpersonal dynamics, communication, diversity of thought, and task efficiency. "
        "Respond with the MBTI type you recommend and a short, 1-2 sentence reason why."
    )

    try:
        print("Calling LLM with prompt:")
        print(prompt)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )

        print("LLM Response:")
        print(response)

        reply = response["choices"][0]["message"]["content"]
        lines = reply.strip().split("\n")
        recommended = lines[0].strip()
        reason = " ".join(lines[1:]).strip() or "Generated with GPT."

        return jsonify({
            "recommendedType": recommended,
            "reason": reason
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
  app.run(threaded=True, port=5001)
