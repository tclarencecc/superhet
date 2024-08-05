from httpx import AsyncClient
import uuid
from typing import Iterable
from enum import Enum

from app.config import Config
from app.util import timestamp, benchmark

_UUID0 = uuid.UUID(int=0).hex

class DatabaseError(Exception): ...

class _Meth(Enum):
    GET = 1
    POST = 2
    PUT = 3
    DEL = 4

async def _http(meth: _Meth, url: str, json: dict | None, client: AsyncClient) -> any:
    if url.startswith("http://") == False:
        # is relative path, build full path here
        url = f"{Config.QDRANT.HOST}/collections/{Config.COLLECTION.replace(" ", "%20")}{url}"

    headers = { "api-key": Config.QDRANT.KEY }
    
    if meth == _Meth.GET:
        res = await client.get(url, headers=headers)
    elif meth == _Meth.POST:
        res = await client.post(url, headers=headers, json=json)
    elif meth == _Meth.PUT:
        headers["Content-Type"] = "application/json"
        res = await client.put(url, headers=headers, json=json)
    elif meth == _Meth.DEL:
        res = await client.delete(url, headers=headers)

    if res.status_code == 200:
        return res.json()["result"]
    else:
        raise DatabaseError(f"Qdrant server returned http error: {res.status_code}")

async def init(client: AsyncClient):
    res = await _http(_Meth.GET, "/exists", None, client)
    
    if res["exists"] == False:
        await _http(_Meth.PUT, "", {
            "vectors": {
                "size": Config.LLAMA.EMBEDDING.SIZE,
                "distance": "Cosine"
            }
        }, client)

        await _http(_Meth.PUT, "/points?wait=true", {
            "points": [{
                "id": _UUID0,
                "vector": [0.0] * Config.LLAMA.EMBEDDING.SIZE,
                "payload": {
                    "sources": []
                }
            }]
        }, client)

@benchmark("db create")
async def create(input: Iterable[dict], src: str, client: AsyncClient) -> bool:
    """
    dict attributes:\n
    len: int - length of list (documents / vectors)\n
    documents: list[str] - list of text chunks\n
    vectors: list[list[float]] - list of vectors (list[float])\n
    """
    count = 0

    while (dv := next(input, None)) is not None:
        points = []
        for i in range(dv["len"]):
            points.append({
                "id": uuid.uuid4().hex,
                "vector": dv["vectors"][i],
                "payload": {
                    "source": src,
                    "document": dv["documents"][i]
                }
            })

        await _http(_Meth.PUT, "/points?wait=true", {
            "points": points
        }, client)

        count += len(points)

    return await _update_journal(src, count, client) # TODO rollback needed if failed?

async def read(vector: list[float], client: AsyncClient) -> str:
    res = await _http(_Meth.POST, "/points/search", {
        "vector": vector,
        "with_payload": True,
        "limit": Config.QDRANT.READ_LIMIT,
        "score_threshold": 0.65
    }, client)

    ret = ""
    for hit in res:
        if hit["payload"].get("document") is not None:
            if ret != "":
                ret += "\n"
            ret += hit["payload"]["document"]

    return ret.strip() # in case some document are actually ""

async def list(client: AsyncClient) -> list[any]:
    """
    returns: [{ name, count, timestamp }]
    """
    res = await _http(_Meth.POST, "/points", {
        "ids": [_UUID0],
        "with_payload": True
    }, client)

    return res[0]["payload"]["sources"]

@benchmark("db delete")
async def delete(src: str, client: AsyncClient) -> bool:
    await _http(_Meth.POST, "/points/delete?wait=true", {
        "filter": {
            "must": [{
                "key": "source",
                "match": {
                    "value": src
                }
            }]
        }
    }, client)

    return await _update_journal(src, -1, client) # TODO rollback needed if failed?
    
async def _update_journal(src: str, count: int, client: AsyncClient) -> bool:
    """
    count: > 0, add entry to journal. duplicate src name is allowed\n
    0 or -int, delete ALL entries with src name
    """
    sources = await list(client)

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

    res = await _http(_Meth.POST, "/points/payload?wait=true", {
        "payload": {
            "sources": sources
        },
        "points": [_UUID0]
    }, client)

    return res["status"] == "completed"

async def drop(collection: str, client: AsyncClient):
    # use full path as collection name is diff!
    await _http(_Meth.DEL, f"{Config.QDRANT.HOST}/collections/{collection}", None, client)
