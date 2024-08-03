from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, VectorParams, Distance
import warnings
import uuid

from app.config import Config
from app.util import benchmark, timestamp

# qdrant raises 'Api key is used with an insecure connection' but theres
# no need for tls as qdrant exists in the same machine as the python app!
warnings.filterwarnings("ignore", module="qdrant_client")

_UUID0 = uuid.UUID(int=0).hex

class _DBClient(object):
    def __init__(self):
        self.client: AsyncQdrantClient = None

    async def __aenter__(self) -> AsyncQdrantClient:
        self.client = AsyncQdrantClient(Config.QDRANT.HOST,
            api_key=Config.QDRANT.KEY,
            grpc_port=Config.QDRANT.GRPC
        )
        return self.client
    
    async def __aexit__(self, type, value, traceback):
        await self.client.close()

async def init():
    async with _DBClient() as client:
        res = await client.collection_exists(Config.COLLECTION)
        if res == False:
            await client.create_collection(Config.COLLECTION, VectorParams(
                size=Config.LLAMA.EMBEDDING.SIZE,
                distance=Distance.COSINE
            ))

            # create the journal entry
            await client.upsert(Config.COLLECTION, [PointStruct(
                id=_UUID0,
                payload={
                    "sources": []
                },
                vector=[0.0] * Config.LLAMA.EMBEDDING.SIZE
            )])

@benchmark("db create")
async def create(documents: list[str], vectors: list[list[float]], src: str) -> bool:
    if len(documents) != len(vectors):
        raise ValueError("db create documents & vectors have different sizes.")

    points = []
    for i in range(0, len(vectors)):
        points.append(PointStruct(
            id=uuid.uuid4().hex,
            payload={
                "source": src,
                "document": documents[i]
            },
            vector=vectors[i]
        ))

    async with _DBClient() as client:
        await client.upsert(Config.COLLECTION, points)

    return await _update_journal(src, len(points)) # TODO rollback needed if failed?

@benchmark("db read")
async def read(vector: list[float]) -> str:
    async with _DBClient() as client:
        hits = await client.search(Config.COLLECTION, vector,
            limit=Config.QDRANT.READ_LIMIT,
            score_threshold=0.65
        )

    ret = ""
    for hit in hits:
        if ret != "":
            ret += "\n"

        if hit.payload.get("document") is not None:
            ret += hit.payload["document"]

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

@benchmark("db drop")
async def drop(collection: str) -> bool:
    async with _DBClient() as client:
        return await client.delete_collection(collection)
