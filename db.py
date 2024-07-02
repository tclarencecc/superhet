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

class _DBClient(object):
    def __init__(self):
        self.client: QdrantClient = None

    def __enter__(self) -> QdrantClient:
        self.client = QdrantClient(config.get(ConfigKey.DB_HOST))
        return self.client
    
    def __exit__(self, type, value, traceback):
        self.client.close()

@benchmark("db create")
def create(collection: str, input: str, src: str, chunk: int, alphabet=True) -> tuple[bool, int]:
    documents = list[str]

    if input.startswith("./"):
        try:
            f = open(input)
        except FileNotFoundError:
            print(input + " not found.")
        else:
            documents = chunker.split(f.read(), chunk, alphabet=alphabet)
        finally:
            f.close()
    elif input.startswith("<!DOCTYPE html>"):
        # TODO
        pass
    else:
        documents = chunker.split(input, chunk, alphabet=alphabet)

    count = len(documents)
    ids = []
    metadata = []

    for _ in range(count):
        ids.append(uuid.uuid4().hex)
        metadata.append({ "source": src })

    with _DBClient() as client:
        # does repeated batch embed + upload of 32 docs per batch
        res = client.add(collection, documents, metadata=metadata, ids=ids)

    # indexing payload.source may be necessary..
    # https://qdrant.tech/documentation/concepts/indexing/

    return (len(res) == count, count)

@benchmark("db read")
def read(collection: str, query: str, limit=1) -> str:
    if query == "":
        raise Exception("Database.read undefined query.")

    with _DBClient() as client:
        hits = client.query(collection, query, limit=limit)

    ret = ""
    for hit in hits:
        if ret != "":
            ret = ret + "\n"
        # score in hit.metadata["score"], if needed
        ret = ret + hit.metadata["document"]

    return ret
    
@benchmark("db delete")
def delete(collection: str, src: str) -> bool:
    with _DBClient() as client:
        res = client.delete(collection, points_selector=Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=src)
            )]
        ))
        return res.status.value == "completed"

@benchmark("db drop")
def drop(collection: str) -> bool:
    with _DBClient() as client:
        return client.delete_collection(collection)
