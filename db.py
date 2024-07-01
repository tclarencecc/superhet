from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import uuid
import chunker
import config
from config import ConfigKey
from util import benchmark
import os

# qdrant fastembed reads from this env-var for embedding model path
os.environ["FASTEMBED_CACHE_PATH"] = config.get(ConfigKey.FASTEMBED_CACHE)

def _client_factory() -> QdrantClient:
    return QdrantClient(config.get(ConfigKey.DB_HOST))
    
@benchmark("db create")
def create(collection: str, input: str, src: str, chunk: int, alphabet=True) -> None:
    documents = list[str]

    if input.startswith("./"):
        try:
            f = open(input)
        except FileNotFoundError:
            print(input + " not found.")
        else:
            documents = chunker.split(f.read(), chunk, alphabet=alphabet)
    elif input.startswith("<!DOCTYPE html>"):
        # TODO
        pass
    else:
        documents = chunker.split(input, chunk, alphabet=alphabet)

    ids = []
    metadata = []
    for doc in documents:
        ids.append(uuid.uuid4().hex)
        metadata.append({ "source": src })

    _client_factory().add(collection, documents, metadata=metadata, ids=ids)

    # indexing payload.source may be necessary..
    # https://qdrant.tech/documentation/concepts/indexing/

@benchmark("db read")
def read(collection: str, query: str, limit=1) -> str:
    if query == "":
        raise Exception("Database.read undefined query.")

    hits = _client_factory().query(collection, query, limit=limit)
    ret = ""
    
    for hit in hits:
        if ret != "":
            ret = ret + "\n"
        # score in hit.metadata["score"], if needed
        ret = ret + hit.metadata["document"]
    return ret
    
@benchmark("db delete")
def delete(collection: str, src: str) -> None:
    _client_factory().delete(collection, points_selector=Filter(
        must=[FieldCondition(
            key="source",
            match=MatchValue(value=src)
        )]
    ))

@benchmark("db drop")
def drop(collection: str) -> None:
    _client_factory().delete_collection(collection)
