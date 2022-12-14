import os

import openai
from flask import Flask, request
from flask_basicauth import BasicAuth

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.getenv("BASIC_AUTH_USERNAME")
app.config['BASIC_AUTH_PASSWORD'] = os.getenv("BASIC_AUTH_PASSWORD")
app.config['BASIC_AUTH_FORCE'] = True
basic_auth = BasicAuth(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        text = request.json["text"]
        response = openai.Completion.create(
            model=os.getenv("MODEL"),
            prompt=text,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text

    return "404 Not Found"

@app.route("/chat_summary", methods=("GET", "POST"))
def chat_summary():
    if request.method == "POST":
        transcript = request.json["transcript"]
        response = openai.Completion.create(
            model=os.getenv("MODEL"),
            prompt=generate_summarize_prompt(transcript),
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text

    return "404 Not Found"

@app.route("/chat_sentiment", methods=("GET", "POST"))
def chat_sentiment():
    if request.method == "POST":
        transcript = request.json["transcript"]
        response = openai.Completion.create(
            model=os.getenv("MODEL"),
            prompt=generate_sentiment_prompt(transcript),
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text

    return "404 Not Found"

@app.route("/autocorrect", methods=("GET", "POST"))
def text_autocorrect():
    if request.method == "POST":
        text = request.json["text"]
        response = openai.Completion.create(
            model=os.getenv("MODEL"),
            prompt=generate_autocorrect_prompt(text),
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text

    return "404 Not Found"

@app.route("/translate", methods=("GET", "POST"))
def translate():
    if request.method == "POST":
        content = request.json["content"]
        toLanguage = request.json["toLanguage"]
        response = openai.Completion.create(
            model=os.getenv("MODEL"),
            prompt=generate_translate_prompt(content, toLanguage),
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text

    return "404 Not Found"

@app.route("/generate_qna", methods=("GET", "POST"))
def qnaGenerator():
    if request.method == "POST":
        topic = request.json["topic"]
        count = request.json["count"]
        response = openai.Completion.create(
            model=os.getenv("MODEL"),
            prompt=generate_qna_prompt(topic, count),
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text

    return "404 Not Found"

@app.route("/generate_variations", methods=("GET", "POST"))
def variationsGenerator():
    if request.method == "POST":
        utterance = request.json["utterance"]
        count = request.json["count"]
        response = openai.Completion.create(
            model=os.getenv("MODEL"),
            prompt=generate_variation_prompt(utterance, count),
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text

    return "404 Not Found"

@app.route("/generate_image", methods=("GET", "POST"))
def imageGenerator():
    if request.method == "POST":
        image = request.json["image"]
        size = request.json["size"]
        response = openai.Image.create(
            prompt=image,
            n=1,
            size=size
        )
        return response["data"][0]['url']

    return "404 Not Found"

def generate_summarize_prompt(transcript):
    return "Summarize this chat: \n {}".format(transcript)

def generate_sentiment_prompt(transcript):
    return "Analyze the sentiment whether it is Positive, Neutral or Negative: \n {}".format(transcript)

def generate_autocorrect_prompt(text):
    return "Autocorrect this: \n {}".format(text)

def generate_translate_prompt(content, language):
    return "Translate to {}: \n {}".format(language, content)

def generate_qna_prompt(topic, count):
    return "Create {} QnAs about {}".format(count, topic)

def generate_variation_prompt(utterance, count):
    return "Provide {} variations of below utterance: \n {}".format(count, utterance)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)