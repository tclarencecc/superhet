from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import Iterable
from config import Config
from util import benchmark, timestamp
import os
import warnings
import uuid

# qdrant fastembed reads from this env-var for embedding model path
os.environ["FASTEMBED_CACHE_PATH"] = Config.FASTEMBED.PATH

# qdrant raises 'Api key is used with an insecure connection' but theres
# no need for tls as qdrant exists in the same machine as the python app!
warnings.filterwarnings("ignore", module="qdrant_client")

_UUID0 = uuid.UUID(int=0).hex

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
            await client.add(Config.COLLECTION, [""],
                metadata=[{ "sources": [] }],
                ids=[_UUID0]
            )

@benchmark("db create")
async def create(documents: Iterable[str], src: str) -> bool:
    class MetaData:
        def __iter__(self):
            return self
        def __next__(self):
            return { "source": src }

    async with _DBClient() as client:
        # does repeated batch embed + upload of 32 docs per batch
        res = await client.add(Config.COLLECTION, documents, metadata=MetaData())

    if len(res) > 0:
        return await _update_journal(src, len(res)) # TODO rollback needed if failed?

    return False

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

@benchmark("db list")
async def list() -> list[any]:
    """
    returns: [{ name, count, timestamp }]
    """
    async with _DBClient() as client:
        res = await client.retrieve(Config.COLLECTION, [_UUID0], with_vectors=False)
        return res[0].payload["sources"]
    
@benchmark("db delete")
async def delete(src: str) -> bool:
    async with _DBClient() as client:
        res = await client.delete(Config.COLLECTION, points_selector=Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=src)
            )]
        ))

    if res.status.value == "completed":
        return await _update_journal(src, -1) # TODO rollback needed if failed?
    
    return False
    
async def _update_journal(src: str, count: int) -> bool:
    """
    count: > 0, add entry to journal. duplicate src name is allowed\n
    0 or -int, delete ALL entries with src name
    """
    async with _DBClient() as client:
        res = await client.retrieve(Config.COLLECTION, [_UUID0], with_vectors=False)
        sources = res[0].payload["sources"]

        if count > 0:
            sources.append({
                "name": src,
                "count": count,
                "timestamp": timestamp()
            })
        else:
            ns = []
            for s in sources:
                if s["name"] != src: # filter out all (duplicated) entries
                    ns.append(s)
            sources = ns

        res = await client.set_payload(Config.COLLECTION,
            payload={ "sources": sources },
            points=[_UUID0]
        )
        return res.status.value == "completed"

# @benchmark("db drop")
# async def drop(collection: str) -> bool:
#     async with _DBClient() as client:
#         return await client.delete_collection(collection)
