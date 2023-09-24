import ast
import concurrent.futures
import os
import pprint
import re
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import openai
import pandas as pd
import tiktoken
from flask import Flask, render_template
from scipy import spatial
from sqlalchemy import create_engine


def bypass_https():
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry

    # Bypass SSL verification
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.verify = False  # THIS IS RISKY; DON'T USE IN PRODUCTION
    requests.sessions.Session.request = session.request


class OpenAIEmbedding:
    def __init__(self, MODEL_NAME: str = "text-embedding-ada-002") -> None:
        self.model_name = MODEL_NAME
        self.encoding = tiktoken.encoding_for_model(model_name=MODEL_NAME)

    def get_tokens(self, text: str):
        tokens = self.encoding.encode(text)
        return tokens

    def get_embedding(self, text: str) -> List[float]:
        response = openai.Embedding.create(model=self.model_name, input=text, api_key=self.api_key)
        return response["data"][0]["embedding"]

    def _get_embeddings_for_batch(self, batch: List[str]) -> List[List[float]]:
        """Helper function to fetch embeddings for a batch."""
        response = openai.Embedding.create(model=self.model_name, input=batch, api_key=self.api_key)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        return [e["embedding"] for e in response["data"]]

    def get_embeddings_parallel(self, texts: List[str]) -> List[List[float]]:
        BATCH_SIZE = 2000  # OpenAI allows us to submit up to 2048 embedding inputs per request

        # Create batches
        batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

        embeddings = []

        # Use ThreadPoolExecutor to fetch data for each batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for batch_result in executor.map(self._get_embeddings_for_batch, batches):
                embeddings.extend(batch_result)

        return embeddings

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        BATCH_SIZE = 1000  # OpenAI allows us to submit up to 2048 embedding inputs per request

        embeddings = []
        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch = texts[batch_start:batch_end]
            print(f"Batch {batch_start} to {batch_end-1}")
            response = openai.Embedding.create(model=self.model_name, input=batch, api_key=self.api_key)
            for i, be in enumerate(response["data"]):
                assert i == be["index"]  # double check embeddings are in same order as input
            batch_embeddings = [e["embedding"] for e in response["data"]]
            embeddings.extend(batch_embeddings)
        return embeddings


class DataCleaningUtils:
    @staticmethod
    def filter_kannada_text(s: str) -> str:
        # Given a string, remove all Kannada characters
        s = re.sub(r"[\u0C80-\u0CFF]", "", s)
        return " ".join(s.split())

    @staticmethod
    def does_have_english_chars(s: str) -> bool:
        # Given a string, return True if it has at least one English character, else False
        return bool(re.search(r"[a-zA-Z]", s))

    @staticmethod
    def split_on_semicolon(s: str) -> list:
        # Given a string, split it on semicolons and return a list of the parts
        parts = s.split(";")
        return [part.strip() for part in parts if part.strip()]


class Dataset:
    def get_data(self):
        df = self._get_data_from_db()
        df = self._clean_data(df)
        return df

    def _get_db_engine(self):
        DATABASE_URL = os.environ.get("DATABASE_URL")
        return create_engine(DATABASE_URL)

    def _get_data_from_db(self) -> pd.DataFrame:
        engine = self._get_db_engine()
        return pd.read_sql("SELECT id, content FROM entries e WHERE lang='english'", con=engine)

    def _clean_data(self, df) -> pd.DataFrame:
        """
        Given a DataFrame with 2 columns named 'id' and 'content', applies a series of transformations to clean the data and returns the cleaned data as dataframe.
        """

        # 1. Apply filter_kannada_text
        df["content"] = df["content"].apply(DataCleaningUtils.filter_kannada_text)

        # 2. Filter out rows where does_have_english_chars is False
        df = df[df["content"].apply(DataCleaningUtils.does_have_english_chars)]

        # 3. Split on semicolon and expand DataFrame
        rows = []
        for _, row in df.iterrows():
            split_contents = DataCleaningUtils.split_on_semicolon(row["content"])
            for content in split_contents:
                rows.append({"id": row["id"], "content": content})

        return pd.DataFrame(rows)


def create_embeddings(df):
    embedding_service = OpenAIEmbedding()
    contents = list(df["content"].array)
    ids = list(df["id"].array)
    tokens = sum(len(embedding_service.get_tokens(content)) for content in contents)
    amount = (tokens / 1000) * 0.0001
    print(f"Total tokens to be used: {tokens}, amount: {amount}$ proceed? ")
    input()
    embeddings = embedding_service.get_embeddings_parallel(contents)
    return pd.DataFrame({"id": ids, "embedding": embeddings, "content": contents})


def create_and_save_embeddings(n: int = 100):
    dataset = Dataset()
    print("Getting data from database...")
    df = dataset.get_data()
    print(f"Creating embeddings for {len(df)} samples...")
    # df = df.sample(n=2000)
    df = create_embeddings(df)
    print("Saving embeddings to csv file...")
    # save it to a csv file embeddings-data.csv
    df.to_csv("embeddings-data-all.csv", index=False)
    print("Done!")


def get_embeddings_from_csv() -> pd.DataFrame:
    df = pd.read_csv("embeddings-data.csv")
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    return df


@dataclass
class Relation:
    id: int
    relation_types: str
    guid: str
    content: str
    relatedness: float


@dataclass
class Entry:
    id: int
    content: str
    lang: str
    phones: str
    guid: str
    relations: List[Relation] = field(default_factory=list)


class Search:
    def __init__(self) -> None:
        self.df = get_embeddings_from_csv()
        self.embedding_service = OpenAIEmbedding()
        DATABASE_URL = os.environ.get("DATABASE_URL")
        self.engine = create_engine(DATABASE_URL)


    def _strings_ranked_by_relatedness(self, query: str, top_n: int = 10) -> Tuple[List[id], List[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding = self.embedding_service.get_embedding(text=query)
        print("Got the embeddings from API")
        relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
        ids_and_relatedness = [
            (row["id"], relatedness_fn(query_embedding, row["embedding"])) for i, row in self.df.iterrows()
        ]
        ids_and_relatedness.sort(key=lambda x: x[1], reverse=True)
        ids, relatednesses = zip(*ids_and_relatedness)
        return ids[:top_n], relatednesses[:top_n]

    def _get_ranked_results_for_query(self, query: str, top_n: int = 10) -> Tuple[List[int], List[float]]:
        """
        Queries db vector store for top_n results for the given query embeddings.
        """
        db_query = """
        select entry_id, 1 - (embedding <=> '{query_embedding}') as relatedness from embeddings order by relatedness desc limit {n}
        """
        query_embedding = self.embedding_service.get_embedding(text=query)
        db_query = db_query.format(query_embedding=query_embedding, n=top_n)
        df = pd.read_sql(db_query, con=self.engine)
        return list(df["entry_id"].array), list(df["relatedness"].array)

    def search(self, query) -> List[Entry]:
        # ids, relatednesses = self._strings_ranked_by_relatedness(query=query, top_n=5)
        ids, relatednesses = self._get_ranked_results_for_query(query=query, top_n=5)
        id_to_relatedness = dict(zip(ids, relatednesses))
        if not ids:
            return []
        db_query = """
        with main_result as (
        select
            r.id,
            r.from_id ,
            r.to_id,
            r."types" ,
            'main' as result_type
        from
            entries e
        inner join relations r on
            r.to_id = e.id
        where
            e.id in ({ids})
        ), 
        other_results as (
        select
            r.id,
            r.from_id ,
            r.to_id,
            r."types" ,
            'other' as result_type
        from
            relations r
        where
            1 = 0 and
            r.from_id in (
            select
                from_id
            from
                main_result)
            and r.id not in (
            select
                id
            from
                main_result) ),
        res as (
        select
            *
        from
            main_result
        union all
        select
            *
        from
            other_results) 
        select 
            res.id relation_id,
            res.from_id,
            res.to_id,
            res.result_type,
            res.types,
            f.content from_lang_content,
            f.lang from_lang,
            f.phones from_lang_phones,
            f.guid from_lang_guid,
            t."content" to_lang_content,
            t.lang to_lang,
            t.guid to_lang_guid
        from
            res
        inner join entries f on
            f.id = res.from_id
        inner join entries t on
            t.id = res.to_id
        """

        db_query = db_query.format(ids=",".join([str(i) for i in ids]))
        df = pd.read_sql(db_query, con=self.engine)
        search_results = df.to_dict(orient="records")

        # The rows are not ordered according to relatedness, the ids correspond to to_id fields in the dict. sort by relatedness. if relatedness is not found for an id, then put it at the end.
        search_results.sort(key=lambda x: id_to_relatedness.get(x["to_id"], -1), reverse=True)

        from_id_to_entry: dict[int, Entry] = {}
        for search_result in search_results:
            if search_result["from_id"] not in from_id_to_entry:
                entry = Entry(
                    id=search_result["from_id"],
                    content=search_result["from_lang_content"],
                    lang=search_result["from_lang"],
                    phones=", ".join(search_result["from_lang_phones"]),
                    guid=search_result["from_lang_guid"],
                )
                from_id_to_entry[search_result["from_id"]] = entry
            from_id_to_entry[search_result["from_id"]].relations.append(
                Relation(
                    id=search_result["relation_id"],
                    relation_types=", ".join(search_result["types"]),
                    guid=search_result["to_lang_guid"],
                    content=search_result["to_lang_content"],
                    relatedness=id_to_relatedness.get(search_result["to_id"], -1),
                )
            )

        return list(from_id_to_entry.values())



app = Flask(__name__)


@app.route("/dictionary/english/kannada/<english_text>")
def search(english_text: str):
    search = Search()
    results = search.search(english_text)
    pprint.pprint([asdict(result) for result in results])
    return render_template("results.html", data={"entries": results, "search_query": english_text})


if __name__ == "__main__":
    bypass_https()  # TODO Temporary, remove it later.
    app.run(debug=True)
    # create_and_save_embeddings()
