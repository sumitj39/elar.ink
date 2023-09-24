import os
from flask import Flask, render_template, request, Response
from embedding import GTEEmbedding, OpenAIEmbedding

import utils
from search import Search, EmbeddingSearchWithCsv, EmbeddingSearchWithDB

from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


app = Flask(__name__)

# Set up Prometheus metrics
metrics = PrometheusMetrics(app)
@app.route("/prometheus/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    
@app.route("/")
def main():
    return "Hello!"  # requests tracked by default


@app.route("/dictionary/english/kannada")
def search():
    # get model query parameter from request:
    english_text = request.args.get("searchword", "")
    if not english_text:
        return render_template("results.html", data={"entries": [], "search_query": "", "model": ""})

    model = request.args.get("model", "")
    if not model:
        model = "gte"
    if model == "gte":
        embedding_service = gte_embedding_service
        # fname = "./embeddings-data-gte-base-sample.csv"
        print("Using GTEEmbedding")
        db_table = "embeddings_gte_base"
    else:
        # By default use OpenAI
        embedding_service = OpenAIEmbedding(api_key=os.environ.get("OPENAI_API_KEY"))
        # fname = "./embeddings-data-openai-vs-gte-base-sample.csv"
        print("Using OpenAIEmbedding")
        db_table = "embeddings"

    # embedding_search_service = EmbeddingSearchWithCsv(
    #     fname=fname, embedding_service=embedding_service
    # )
    embedding_search_service = EmbeddingSearchWithDB(
        embedding_service=embedding_service, db_table=db_table
    )  # TODO clean up this dependency injection mess
    search = Search(embedding_search_service=embedding_search_service)
    results = search.search(english_text)
    with utils.timed("render_template_timer"):
        tmpl = render_template("results.html", data={"entries": results, "search_query": english_text, "model": model})
    return tmpl


@app.route("/api/v1/embeddings")
def get_embeddings():
    model = request.args.get("model", "")
    if not model:
        model = "gte"

    text = request.args.get("text", "")

    if model == "gte":
        model_name = "./models/thenlper_gte-base/" # TODO Refactor this dependency injection mess
        embedding_service = GTEEmbedding(model_name=model_name)
    elif model == "openai":
        # By default use OpenAI
        embedding_service = OpenAIEmbedding(api_key=os.environ.get("OPENAI_API_KEY"))
    embedding = embedding_service.get_embedding(text=text)
    return {"model": f"{model}::{model_name}", "data": {"embedding": embedding}}


if __name__ == "__main__":
    # bypass_https()  # TODO Temporary, remove it later.
    model_name = "./models/thenlper_gte-base/"
    gte_embedding_service = GTEEmbedding(model_name=model_name)
    app.run(debug=True)
