from typing import List

import openai
import tiktoken
from scipy import spatial
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import utils


def cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)


class EmbeddingBase:
    def get_tokens(self, text: str):
        raise NotImplementedError

    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingBase):
    def __init__(
        self, model_name: str = "text-embedding-ada-002", api_key: str = ""
    ) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name=model_name)
        self.api_key = api_key
        utils.bypass_https()

    def get_tokens(self, text: str):
        tokens = self.encoding.encode(text)
        return tokens

    def get_embedding(self, text: str) -> List[float]:
        response = openai.Embedding.create(
            model=self.model_name, input=text, api_key=self.api_key
        )
        return response["data"][0]["embedding"]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        print("Getting embeddings for texts...")
        return self.__get_embeddings_parallel(texts=texts)

    def __get_embeddings_for_batch(self, batch: List[str]) -> List[List[float]]:
        """Helper function to fetch embeddings for a batch."""
        response = openai.Embedding.create(
            model=self.model_name, input=batch, api_key=self.api_key
        )
        for i, be in enumerate(response["data"]):
            assert (
                i == be["index"]
            )  # double check embeddings are in same order as input
        return [e["embedding"] for e in response["data"]]

    def __get_embeddings_parallel(self, texts: List[str]) -> List[List[float]]:
        BATCH_SIZE = (
            2000  # OpenAI allows us to submit up to 2048 embedding inputs per request
        )

        # Create batches
        batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

        embeddings = []

        import concurrent.futures

        # Use ThreadPoolExecutor to fetch data for each batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for batch_result in executor.map(self.__get_embeddings_for_batch, batches):
                embeddings.extend(batch_result)

        return embeddings


class GTEEmbedding(EmbeddingBase):
    """Open source GTE model provides similar performance to OpenAI's proprietary models, with much smaller vector sizes (384 vs 1536)"""

    def __init__(self, model_name: str = "thenlper/gte-small") -> None:
        self.model_name = model_name
        print(f"Using model name: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = SentenceTransformer(self.model_name)

    def get_tokens(self, text: str):
        tokens = self.tokenizer(text)
        return tokens["input_ids"]

    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def _encode_batch(self, batch: List[str]) -> List[List[float]]:
        # This is a helper function to encode a batch of texts
        print(f"Encoding batch of {len(batch)} texts...")
        return self.model.encode(batch).tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        from multiprocessing import Pool, cpu_count

        batch_size = 2000
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        # Limit the workers to 6 or available CPUs, whichever is less
        max_workers = min(6, cpu_count())

        with Pool(max_workers) as pool:
            results = pool.map(self._encode_batch, batches)

        # Flatten the results
        print("Flattening results...")
        embeddings = [
            embedding for batch_result in results for embedding in batch_result
        ]
        return embeddings
