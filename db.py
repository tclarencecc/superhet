from qdrant_client import AsyncQdrantClient
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
        self.client: AsyncQdrantClient = None

    async def __aenter__(self) -> AsyncQdrantClient:
        self.client = AsyncQdrantClient(config.get(ConfigKey.DB_HOST))
        return self.client
    
    async def __aexit__(self, type, value, traceback):
        await self.client.close()

@benchmark("db create")
async def create(collection: str, input: str, src: str, chunk: int, alphabet=True) -> tuple[bool, int]:
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

    class MetaData:
        def __iter__(self):
            return self
        def __next__(self):
            return { "source": src }

    async with _DBClient() as client:
        # does repeated batch embed + upload of 32 docs per batch
        res = await client.add(collection, documents, metadata=MetaData())

    # indexing payload.source may be necessary..
    # https://qdrant.tech/documentation/concepts/indexing/

    return (len(res) == count, count)

@benchmark("db read")
async def read(collection: str, query: str, limit=1) -> str:
    if query == "":
        raise Exception("Database.read undefined query.")

    async with _DBClient() as client:
        hits = await client.query(collection, query, limit=limit)

    ret = ""
    for hit in hits:
        if ret != "":
            ret += "\n"
        # score in hit.metadata["score"], if needed
        ret += hit.metadata["document"]

    return ret
    
@benchmark("db delete")
async def delete(collection: str, src: str) -> bool:
    async with _DBClient() as client:
        res = await client.delete(collection, points_selector=Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=src)
            )]
        ))
        return res.status.value == "completed"

@benchmark("db drop")
async def drop(collection: str) -> bool:
    async with _DBClient() as client:
        return await client.delete_collection(collection)
