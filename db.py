from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import Iterable
from config import Config
from util import benchmark
import os
import warnings
import uuid

# qdrant fastembed reads from this env-var for embedding model path
os.environ["FASTEMBED_CACHE_PATH"] = Config.FASTEMBED.PATH

# qdrant raises 'Api key is used with an insecure connection' but theres
# no need for tls as qdrant exists in the same machine as the python app!
warnings.filterwarnings("ignore", module="qdrant_client")

class _DBClient(object):
    def __init__(self):
        self.client: AsyncQdrantClient = None

    async def __aenter__(self) -> AsyncQdrantClient:
        self.client = AsyncQdrantClient(Config.QDRANT.HOST, api_key=Config.QDRANT.KEY)
        return self.client
    
    async def __aexit__(self, type, value, traceback):
        await self.client.close()

async def init():
    async with _DBClient() as client:
        res = await client.collection_exists(Config.COLLECTION)
        if res == False:
            # create the journal entry
            await client.add(Config.COLLECTION, [""], metadata=[{}], ids=[uuid.UUID(int=0).hex])

@benchmark("db create")
async def create(documents: Iterable[str], src: str) -> int:
    class MetaData:
        def __iter__(self):
            return self
        def __next__(self):
            return { "source": src }

    async with _DBClient() as client:
        # does repeated batch embed + upload of 32 docs per batch
        res = await client.add(Config.COLLECTION, documents, metadata=MetaData())

    # indexing payload.source may be necessary..
    # https://qdrant.tech/documentation/concepts/indexing/

    return len(res)

@benchmark("db read")
async def read(query: str, limit=1) -> str:
    if query == "":
        raise ValueError("db.read undefined query.")

    async with _DBClient() as client:
        hits = await client.query(Config.COLLECTION, query, limit=limit)

    ret = ""
    for hit in hits:
        if ret != "":
            ret += "\n"
        if hit.metadata.get("document") is not None:
            ret += hit.metadata["document"]

    return ret.strip()
    
@benchmark("db delete")
async def delete(src: str) -> bool:
    async with _DBClient() as client:
        res = await client.delete(Config.COLLECTION, points_selector=Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=src)
            )]
        ))
        return res.status.value == "completed"

# @benchmark("db drop")
# async def drop(collection: str) -> bool:
#     async with _DBClient() as client:
#         return await client.delete_collection(collection)
