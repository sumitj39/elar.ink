import os
import re

import pandas as pd

import utils
from embedding import GTEEmbedding, OpenAIEmbedding


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

    def _get_data_from_db(self) -> pd.DataFrame:
        engine = utils.get_db_engine()
        return pd.read_sql(
            "SELECT id, content FROM entries e WHERE lang='english'", con=engine
        )

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


def create_embeddings(df, embedding_provider: str, model_name: str):
    if embedding_provider == "gte":
        embedding_service = GTEEmbedding(
            model_name=model_name
        )  # TODO find a better way to send model name
        print("Using GTEEmbedding")
    else:
        # By default use OpenAI
        embedding_service = OpenAIEmbedding(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        print("Using OpenAIEmbedding")

    contents = list(df["content"].array)
    ids = list(df["id"].array)
    tokens = sum(len(embedding_service.get_tokens(content)) for content in contents)
    amount = (tokens / 1000) * 0.0001
    print(f"Total tokens to be used: {tokens}, amount: {amount}$ proceed? ")
    # input()
    embeddings = embedding_service.get_embeddings(contents)
    return pd.DataFrame({"id": ids, "embedding": embeddings, "content": contents})


def create_and_save_embeddings(n: int = 100):
    dataset = Dataset()
    print("Getting data from database...")
    df = dataset.get_data()
    print(f"Creating embeddings for {len(df)} samples...")
    df = df.sample(n=n)

    df_gte = create_embeddings(
        df, embedding_provider="gte", model_name="./models/thenlper_gte-base/"
    )
    fname = "embeddings-data-gte-base-all.csv"
    print(f"Saving embeddings to csv file... {fname}")
    # save it to a csv file embeddings-data.csv
    df_gte.to_csv(fname, index=False)

    # df_oai = create_embeddings(df, embedding_provider="openai", model_name="")
    # fname = "embeddings-data-openai-vs-gte-base-sample.csv"
    # print(f"Saving embeddings to csv file... {fname}")
    # save it to a csv file embeddings-data.csv
    # df_oai.to_csv(fname, index=False)
    print("Done!")


if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"
    create_and_save_embeddings(n=1000)
