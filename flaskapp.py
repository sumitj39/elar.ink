import logging

from flask import Flask, Response, render_template, request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_flask_exporter import PrometheusMetrics

import utils
from embedding import GTEEmbedding, OpenAIEmbedding
from search import EmbeddingSearchWithDB, Search

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 300  # 5 minutes

# Set up logging and metrics
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("flaskapp.log")
file_handler.setFormatter(formatter)

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Set up Prometheus metrics
metrics = PrometheusMetrics(app)


@app.route("/prometheus/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/pages/about")
def about():
    return render_template("about.html")


@app.route("/dictionary/english/kannada")
def search():
    english_text = request.args.get("searchword", "")
    if not english_text:
        return render_template("results.html", data={"entries": [], "search_query": ""})
    app.logger.info(f"search word: {english_text}")

    # Autowire dependencies
    embedding_search_service = EmbeddingSearchWithDB(
        embedding_service=gte_embedding_service, db_table="embeddings"
    )
    search = Search(embedding_search_service=embedding_search_service)

    results = search.search(english_text)

    with utils.timed("render_template_timer"):
        tmpl = render_template(
            "results.html",
            data={"entries": results, "search_query": english_text},
        )
    return tmpl


@app.route("/api/v1/embeddings")
def get_embeddings():
    text = request.args.get("text", "")
    embedding_service = gte_embedding_service

    embedding = embedding_service.get_embedding(text=text)
    return {"model": f"gte::{MODEL_NAME}", "data": {"embedding": embedding}}


### Load model
MODEL_NAME = "thenlper/gte-small"
gte_embedding_service = GTEEmbedding(model_name=MODEL_NAME)
app.logger.info("Loaded the model")

if __name__ == "__main__":
    model_name = "./models/thenlper_gte-base/"
    gte_embedding_service = GTEEmbedding(model_name=model_name)

    # If running as a flask app locally, run it in debug mode. For production, use gunicorn.
    app.run(debug=True)
