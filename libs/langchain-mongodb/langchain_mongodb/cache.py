"""LangChain MongoDB Caches."""

import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Union

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from langchain_mongodb.utils import DRIVER_METADATA
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

logger = logging.getLogger(__file__)


class MongoDBCache(BaseCache):
    """MongoDB Atlas cache

    A cache that uses MongoDB Atlas as a backend
    """

    PROMPT = "prompt"
    LLM = "llm"
    RETURN_VAL = "return_val"

    def __init__(
        self,
        connection_string: str,
        collection_name: str = "default",
        database_name: str = "default",
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize Atlas Cache. Creates collection on instantiation

        Args:
            collection_name (str): Name of collection for cache to live.
                Defaults to "default".
            connection_string (str): Connection URI to MongoDB Atlas.
                Defaults to "default".
            database_name (str): Name of database for cache to live.
                Defaults to "default".
        """
        self.client = _generate_mongo_client(connection_string)
        self.__database_name = database_name
        self.__collection_name = collection_name
        self.chunk_collection = self.database[f"{self.__collection_name}_chunks"]

        if self.__collection_name not in self.database.list_collection_names():
            self.database.create_collection(self.__collection_name)
            # Create an index on key and llm_string
            self.collection.create_index([self.PROMPT, self.LLM])

    @property
    def database(self) -> Database:
        """Returns the database used to store cache values."""
        return self.client[self.__database_name]

    @property
    def collection(self) -> Collection:
        """Returns the collection used to store cache values."""
        return self.database[self.__collection_name]

    def close(self) -> None:
        """Close the MongoClient used by the MongoDBCache."""
        self.client.close()

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        main_doc = (
            self.collection.find_one(self._generate_keys(prompt, llm_string)) or {}
        )

        if not main_doc:
            return None

        if not main_doc.get("is_chunked"):
            return_val = main_doc.get(self.RETURN_VAL)
            return _loads_generations(return_val) if return_val else None

        chunk_key = main_doc.get("chunk_key")
        num_chunks = main_doc.get("num_chunks")
        if not chunk_key or not num_chunks:
            return None

        chunk_keys = [f"{chunk_key}_part_{i + 1}" for i in range(num_chunks)]
        chunk_docs_cursor = self.chunk_collection.find({"_id": {"$in": chunk_keys}})
        docs_by_id = {doc["_id"]: doc["value"] for doc in chunk_docs_cursor}

        if len(docs_by_id) != num_chunks:
            return None

        reassembled = "".join(docs_by_id[key] for key in chunk_keys)
        return _loads_generations(reassembled)  # type: ignore

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        from bson import ObjectId

        serialized_val = _dumps_generations(return_val)
        chunk_size = 800 * 1024

        keys = self._generate_keys(prompt, llm_string)

        if len(serialized_val.encode("utf-8")) <= chunk_size:
            self.collection.update_one(
                keys,
                {"$set": {self.RETURN_VAL: serialized_val, "is_chunked": False}},
                upsert=True,
            )
            return

        chunk_key = str(ObjectId())
        chunks = [
            serialized_val[i : i + chunk_size]
            for i in range(0, len(serialized_val), chunk_size)
        ]
        chunk_docs = [
            {"_id": f"{chunk_key}_part_{i + 1}", "value": chunk}
            for i, chunk in enumerate(chunks)
        ]
        self.chunk_collection.insert_many(chunk_docs)

        self.collection.update_one(
            keys,
            {
                "$set": {
                    "is_chunked": True,
                    "chunk_key": chunk_key,
                    "num_chunks": len(chunks),
                }
            },
            upsert=True,
        )

    def _generate_keys(self, prompt: str, llm_string: str) -> Dict[str, str]:
        """Create keyed fields for caching layer"""
        return {self.PROMPT: prompt, self.LLM: llm_string}

    def clear(self, **kwargs: Any) -> None:
        """Clear cache that can take additional keyword arguments.
        Any additional arguments will propagate as filtration criteria for
        what gets deleted.

        E.g.
            # Delete only entries that have llm_string as "fake-model"
            self.clear(llm_string="fake-model")
        """
        self.collection.delete_many({**kwargs})


class MongoDBAtlasSemanticCache(BaseCache, MongoDBAtlasVectorSearch):
    """MongoDB Atlas Semantic cache.

    A Cache backed by a MongoDB Atlas server with vector-store support
    """

    LLM = "llm_string"
    RETURN_VAL = "return_val"

    def __init__(
        self,
        connection_string: str,
        embedding: Embeddings,
        collection_name: str = "default",
        database_name: str = "default",
        index_name: str = "default",
        wait_until_ready: Optional[float] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Dict[str, Any],
    ):
        """
        Initialize Atlas VectorSearch Cache.
        Assumes collection exists before instantiation

        Args:
            connection_string (str): MongoDB URI to connect to MongoDB Atlas cluster.
            embedding (Embeddings): Text embedding model to use.
            collection_name (str): MongoDB Collection to add the texts to.
                Defaults to "default".
            database_name (str): MongoDB Database where to store texts.
                Defaults to "default".
            index_name: Name of the Atlas Search index.
                defaults to 'default'
            wait_until_ready (float): Wait this time for Atlas to finish indexing
                the stored text. Defaults to None.
        """
        client = _generate_mongo_client(connection_string)
        self.collection = client[database_name][collection_name]
        self.chunk_collection = self.collection.database[
            f"{self.collection.name}_chunks"
        ]
        self.score_threshold = score_threshold
        self._wait_until_ready = wait_until_ready
        super().__init__(
            collection=self.collection,
            embedding=embedding,
            index_name=index_name,
            **kwargs,  # type: ignore
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        post_filter_pipeline = (
            [{"$match": {"score": {"$gte": self.score_threshold}}}]
            if self.score_threshold
            else None
        )

        search_response = self.similarity_search_with_score(
            prompt,
            1,
            pre_filter={self.LLM: {"$eq": llm_string}},
            post_filter_pipeline=post_filter_pipeline,
        )
        if search_response:
            metadata = search_response[0][0].metadata
            if not metadata.get("is_chunked"):
                return_val = metadata.get(self.RETURN_VAL)
                response = _loads_generations(return_val) or return_val  # type: ignore
                return response

            chunk_key = metadata.get("chunk_key")
            num_chunks = metadata.get("num_chunks")
            if not chunk_key or not num_chunks:
                return None

            chunk_keys = [f"{chunk_key}_part_{i + 1}" for i in range(num_chunks)]
            chunk_docs_cursor = self.chunk_collection.find(
                {"_id": {"$in": chunk_keys}}
            )
            docs_by_id = {doc["_id"]: doc["value"] for doc in chunk_docs_cursor}

            if len(docs_by_id) != num_chunks:
                return None

            reassembled = "".join(docs_by_id[key] for key in chunk_keys)
            response = _loads_generations(reassembled) or reassembled  # type: ignore
            return response
        return None

    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: RETURN_VAL_TYPE,
        wait_until_ready: Optional[float] = None,
    ) -> None:
        """Update cache based on prompt and llm_string."""
        from bson import ObjectId

        serialized_val = _dumps_generations(return_val)
        chunk_size = 800 * 1024

        if len(serialized_val.encode("utf-8")) <= chunk_size:
            metadata = {
                self.LLM: llm_string,
                self.RETURN_VAL: serialized_val,
                "is_chunked": False,
            }
        else:
            chunk_key = str(ObjectId())
            chunks = [
                serialized_val[i : i + chunk_size]
                for i in range(0, len(serialized_val), chunk_size)
            ]
            chunk_docs = [
                {"_id": f"{chunk_key}_part_{i + 1}", "value": chunk}
                for i, chunk in enumerate(chunks)
            ]
            self.chunk_collection.insert_many(chunk_docs)
            metadata = {
                self.LLM: llm_string,
                "is_chunked": True,
                "chunk_key": chunk_key,
                "num_chunks": len(chunks),
            }

        self.add_texts([prompt], [metadata])
        wait = self._wait_until_ready if wait_until_ready is None else wait_until_ready

        def is_indexed() -> bool:
            return self.lookup(prompt, llm_string) == return_val

        if wait:
            _wait_until(is_indexed, return_val, timeout=wait)

    def clear(self, **kwargs: Any) -> None:
        """Clear cache that can take additional keyword arguments.
        Any additional arguments will propagate as filtration criteria for
        what gets deleted. It will delete any locally cached content regardless

        E.g.
            # Delete only entries that have llm_string as "fake-model"
            self.clear(llm_string="fake-model")
        """
        self.collection.delete_many({**kwargs})


def _generate_mongo_client(connection_string: str) -> MongoClient:
    return MongoClient(
        connection_string,
        driver=DRIVER_METADATA,
    )


def _dumps_generations(generations: RETURN_VAL_TYPE) -> str:
    """
    Serialization for generic RETURN_VAL_TYPE, i.e. sequence of `Generation`

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: a single string representing a list of generations.

    This, and "_dumps_generations" are duplicated in this utility
    from modules: "libs/community/langchain_community/cache.py"

    This function and its counterpart rely on
    the dumps/loads pair with Reviver, so are able to deal
    with all subclasses of Generation.

    Each item in the list can be `dumps`ed to a string,
    then we make the whole list of strings into a json-dumped.
    """
    return json.dumps([dumps(_item) for _item in generations])


def _loads_generations(generations_str: str) -> Union[RETURN_VAL_TYPE, None]:
    """
    Deserialization of a string into a generic RETURN_VAL_TYPE
    (i.e. a sequence of `Generation`).

    Args:
        generations_str (str): A string representing a list of generations.

    Returns:
        RETURN_VAL_TYPE: A list of generations.


    This function and its counterpart rely on
    the dumps/loads pair with Reviver, so are able to deal
    with all subclasses of Generation.

    See `_dumps_generations`, the inverse of this function.

    Compatible with the legacy cache-blob format
    Does not raise exceptions for malformed entries, just logs a warning
    and returns none: the caller should be prepared for such a cache miss.


    """
    try:
        generations = [loads(_item_str) for _item_str in json.loads(generations_str)]
        return generations
    except (json.JSONDecodeError, TypeError):
        # deferring the (soft) handling to after the legacy-format attempt
        pass

    try:
        gen_dicts = json.loads(generations_str)
        # not relying on `_load_generations_from_json` (which could disappear):
        generations = [Generation(**generation_dict) for generation_dict in gen_dicts]
        logger.warning(
            f"Legacy 'Generation' cached blob encountered: '{generations_str}'"
        )
        return generations
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            f"Malformed/unparsable cached blob encountered: '{generations_str}'"
        )
        return None


def _wait_until(
    predicate: Callable, success_description: Any, timeout: float = 10.0
) -> None:
    """Wait up to 10 seconds (by default) for predicate to be true.

    E.g.:

        wait_until(lambda: client.primary == ('a', 1),
                'connect to the primary')

    If the lambda-expression isn't true after 10 seconds, we raise
    AssertionError("Didn't ever connect to the primary").

    Returns the predicate's first true value.
    """
    start = time.time()
    interval = min(float(timeout) / 100, 0.1)
    while True:
        retval = predicate()
        if retval:
            return retval

        if time.time() - start > timeout:
            raise TimeoutError(f"Didn't ever {success_description}")

        time.sleep(interval)
