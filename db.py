from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import Iterable
import config
from config import ConfigKey
from util import benchmark
import os

# qdrant fastembed reads from this env-var for embedding model path
os.environ["FASTEMBED_CACHE_PATH"] = config.get(ConfigKey.FASTEMBED)["path"]

class _DBClient(object):
    def __init__(self):
        self.client: AsyncQdrantClient = None

    async def __aenter__(self) -> AsyncQdrantClient:
        self.client = AsyncQdrantClient(config.get(ConfigKey.DB_HOST))
        return self.client
    
    async def __aexit__(self, type, value, traceback):
        await self.client.close()

@benchmark("db create")
async def create(collection: str, documents: Iterable[str], src: str) -> int:
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

    return len(res)

@benchmark("db read")
async def read(collection: str, query: str, limit=1) -> str:
    if query == "":
        raise ValueError("db.read undefined query.")

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
