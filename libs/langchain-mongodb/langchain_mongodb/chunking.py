"""Shared utilities for chunking large MongoDB documents.

MongoDB has a 16MB document size limit. When serialized data exceeds a threshold,
we split it into chunks stored in a sibling `_chunks` collection, and store a
pointer document in the main collection.

Chunk documents use _id values like "{chunk_key}_part_{N}" where chunk_key is a
unique ObjectId string and N is 1-indexed.
"""

import logging
from typing import Any, Literal, Optional, Union, overload

from bson import ObjectId
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

# Chunk at 800KB to leave room for metadata fields and BSON overhead
# relative to MongoDB's 16MB document size limit.
CHUNK_SIZE_BYTES = 800 * 1024


def get_chunk_collection_name(collection_name: str) -> str:
    """Return the name for the chunk collection associated with a main collection."""
    return f"{collection_name}_chunks"


def should_chunk(data: Union[bytes, str]) -> bool:
    """Check if data exceeds the chunking threshold.

    For strings, we check the UTF-8 encoded byte length.
    For bytes, we check the raw length.
    """
    if isinstance(data, str):
        return len(data.encode("utf-8")) > CHUNK_SIZE_BYTES
    return len(data) > CHUNK_SIZE_BYTES


def create_chunks(
    data: Union[bytes, str],
    chunk_collection: Collection,
) -> dict[str, Any]:
    """Split data into chunks, store them, and return pointer metadata.

    Args:
        data: The data to chunk (bytes or str).
        chunk_collection: The MongoDB collection to store chunks in.

    Returns:
        A dict with chunking metadata to store in the main document:
        {"is_chunked": True, "chunk_key": str, "num_chunks": int}
    """
    chunk_key = str(ObjectId())

    chunks: list[Union[bytes, str]]
    if isinstance(data, str):
        encoded = data.encode("utf-8")
        raw_chunks = [
            encoded[i : i + CHUNK_SIZE_BYTES]
            for i in range(0, len(encoded), CHUNK_SIZE_BYTES)
        ]
        # Store string chunks as decoded strings
        chunks = [c.decode("utf-8", errors="replace") for c in raw_chunks]
    else:
        chunks = [
            data[i : i + CHUNK_SIZE_BYTES]
            for i in range(0, len(data), CHUNK_SIZE_BYTES)
        ]

    chunk_docs = [
        {"_id": f"{chunk_key}_part_{i + 1}", "value": chunk}
        for i, chunk in enumerate(chunks)
    ]
    chunk_collection.insert_many(chunk_docs)

    return {
        "is_chunked": True,
        "chunk_key": chunk_key,
        "num_chunks": len(chunks),
    }


@overload
def load_chunked_data(
    doc: dict[str, Any],
    data_field: str,
    chunk_collection: Collection,
    is_bytes: Literal[True] = ...,
) -> Optional[bytes]: ...


@overload
def load_chunked_data(
    doc: dict[str, Any],
    data_field: str,
    chunk_collection: Collection,
    is_bytes: Literal[False],
) -> Optional[str]: ...


def load_chunked_data(
    doc: dict[str, Any],
    data_field: str,
    chunk_collection: Collection,
    is_bytes: bool = True,
) -> Optional[Union[bytes, str]]:
    """Load data from a document, reassembling from chunks if necessary.

    Args:
        doc: The document from the main collection.
        data_field: The field name that contains the data (or chunking metadata).
        chunk_collection: The collection where chunks are stored.
        is_bytes: If True, join chunks as bytes. If False, join as strings.

    Returns:
        The reassembled data, or None if chunk reassembly fails.
        If the document is not chunked, returns the data field directly.
    """
    if not doc.get("is_chunked"):
        return doc.get(data_field)

    chunk_key = doc.get("chunk_key")
    num_chunks = doc.get("num_chunks")

    if not chunk_key or not num_chunks:
        logger.warning(
            "Document marked as chunked but missing chunk_key or num_chunks: %s",
            doc.get("_id"),
        )
        return None

    chunk_keys = [f"{chunk_key}_part_{i + 1}" for i in range(num_chunks)]
    chunk_docs = chunk_collection.find({"_id": {"$in": chunk_keys}})
    docs_by_id = {d["_id"]: d["value"] for d in chunk_docs}

    if len(docs_by_id) != num_chunks:
        logger.warning(
            "Chunk count mismatch for chunk_key=%s: expected %d, found %d",
            chunk_key,
            num_chunks,
            len(docs_by_id),
        )
        return None

    if is_bytes:
        return b"".join(docs_by_id[key] for key in chunk_keys)
    else:
        return "".join(docs_by_id[key] for key in chunk_keys)


def delete_chunks(chunk_collection: Collection, chunk_key: str) -> None:
    """Delete all chunk documents for a given chunk_key."""
    # chunk IDs follow the pattern "{chunk_key}_part_{N}"
    chunk_collection.delete_many({"_id": {"$regex": f"^{chunk_key}_part_"}})


def delete_chunks_for_documents(
    main_collection: Collection,
    chunk_collection: Collection,
    query: dict[str, Any],
) -> None:
    """Delete chunks for all chunked documents matching a query.

    This should be called BEFORE deleting the main documents so we can
    find which ones are chunked.
    """
    chunked_docs = main_collection.find(
        {**query, "is_chunked": True},
        {"chunk_key": 1},
    )
    chunk_keys = [doc["chunk_key"] for doc in chunked_docs if doc.get("chunk_key")]
    if chunk_keys:
        for ck in chunk_keys:
            # We don't know num_chunks, so use regex to match all parts
            chunk_collection.delete_many({"_id": {"$regex": f"^{ck}_part_"}})
