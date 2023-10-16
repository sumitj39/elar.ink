import ast
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import openai
import pandas as pd
import tiktoken
from flask import Flask, render_template
from scipy import spatial

import utils
from embedding import GTEEmbedding, OpenAIEmbedding


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


class EmbeddingSearchBase:
    pass


class EmbeddingSearchWithCsv(EmbeddingSearchBase):
    def __init__(self, fname: str, embedding_service) -> None:
        self.df = self.__get_embeddings_from_csv(fname=fname)
        self.embedding_service = embedding_service

    def get_results(self, query: str, top_n: int = 10) -> Tuple[List[int], List[float]]:
        query_embedding = self.embedding_service.get_embedding(text=query)

        relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
        ids_and_relatedness = [
            (row["id"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in self.df.iterrows()
        ]
        ids_and_relatedness.sort(key=lambda x: x[1], reverse=True)
        ids, relatednesses = zip(*ids_and_relatedness)

        return ids[:top_n], relatednesses[:top_n]

    def __get_embeddings_from_csv(self, fname: str) -> pd.DataFrame:
        df = pd.read_csv(fname)
        df["embedding"] = df["embedding"].apply(ast.literal_eval)
        return df


class EmbeddingSearchWithDB(EmbeddingSearchBase):
    def __init__(self, embedding_service, db_table) -> None:
        self.embedding_service = embedding_service
        self.db_table = db_table
        self.engine = utils.get_db_engine()

    def get_results(self, query: str, top_n: int = 10) -> Tuple[List[int], List[float]]:
        """
        Queries db vector store for top_n results for the given query embeddings.
        """
        query_embedding = self.embedding_service.get_embedding(text=query)

        db_query = """
        select entry_id, (embedding <=> '{query_embedding}') as cosine_distance from {db_table} order by cosine_distance limit {n}
        """
        # TODO use placeholders instead of string formatting

        db_query = db_query.format(
            query_embedding=query_embedding, n=top_n, db_table=self.db_table
        )
        with utils.timed("embeddings_search_get_results_timer"):
            df = pd.read_sql(db_query, con=self.engine)

        # cosine_distance contains the distance between 2 Vectors, apply cosine similarity formula to get relatedness
        df["relatedness"] = 1 - df["cosine_distance"]

        return list(df["entry_id"].array), list(df["relatedness"].array)


class Search:
    def __init__(self, embedding_search_service) -> None:
        self.embeddings_search_service = embedding_search_service
        self.engine = utils.get_db_engine()

    def search(self, query) -> List[Entry]:
        # ids, relatednesses = self._strings_ranked_by_relatedness(query=query, top_n=5)
        ids, relatednesses = self.embeddings_search_service.get_results(
            query=query, top_n=10
        )

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

        with utils.timed("search_entries_timer"):
            df = pd.read_sql(db_query, con=self.engine)

        search_results = df.to_dict(orient="records")

        # The rows are not ordered according to relatedness, the ids correspond to to_id fields in the dict. sort by relatedness. if relatedness is not found for an id, then put it at the end.
        # TODO offload the sorting to the db query itself
        id_to_relatedness = dict(zip(ids, relatednesses))
        search_results.sort(
            key=lambda x: id_to_relatedness.get(x["to_id"], -1), reverse=True
        )

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
