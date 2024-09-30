from httpx import AsyncClient, ConnectError
import uuid
from typing import Iterable
from enum import Enum
import sys
import asyncio
import subprocess
import shlex
import signal

from app.config import Config
from app.util import timestamp
from app.decorator import benchmark

_UUID0 = uuid.UUID(int=0).hex

class DbError(Exception): ...

class Db:
    _instance = None

    class Meth(Enum):
        GET = 1
        POST = 2
        PUT = 3
        DEL = 4

    def __new__(cls, client: AsyncClient):
        if cls._instance is None:
            cls._instance = super(Db, cls).__new__(cls)

        return cls._instance
    
    def __init__(self, client: AsyncClient):
        self._proc = None
        self._client = client

    def start(self):
        stdout = None
        stderr = None
        if Config.PROCESS_STDOUT == False:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        self._proc = subprocess.Popen(
            shlex.split(Config.QDRANT.SHELL),
            cwd=Config.QDRANT.PATH,
            env=Config.QDRANT.ENV,
            stdout=stdout,
            stderr=stderr
        )

    def stop(self):
        # KeyboardInterrupt is catched as CancelledError inside asyncio
        # just sigint when exiting regardless of cause
        self._proc.send_signal(signal.SIGINT)
        if self._proc.wait() != 0:
            print(f"pid {self._proc.pid} did not terminate.")

    # implement both enter & exit as async since Db will be "with" together with AsyncClient
    async def __aenter__(self):
        self.start()

    async def __aexit__(self, type, value, traceback):
        self.stop()

    @staticmethod
    async def http(meth: Meth, url: str, json: dict | None) -> any:
        if Db._instance is None:
            raise RuntimeError("Db object uninitialized.")
        
        client = Db._instance._client

        if url.startswith("http://") == False:
            # is relative path, build full path here
            url = f"{Config.QDRANT.HOST}/collections/{Config.COLLECTION.replace(" ", "%20")}{url}"

        headers = { "api-key": Config.QDRANT.KEY }
        
        if meth == Db.Meth.GET:
            res = await client.get(url, headers=headers)
        elif meth == Db.Meth.POST:
            res = await client.post(url, headers=headers, json=json)
        elif meth == Db.Meth.PUT:
            headers["Content-Type"] = "application/json"
            res = await client.put(url, headers=headers, json=json)
        elif meth == Db.Meth.DEL:
            res = await client.delete(url, headers=headers)

        if res.status_code == 200:
            return res.json()["result"]
        else:
            raise DbError(f"Qdrant server returned http error: {res.status_code}")


async def init():
    for i in range(5): # 5 retries at 1 sec interval
        try:
            res = await Db.http(Db.Meth.GET, "/exists", None)
            break
        except ConnectError:
            if i == 4:
                print("Unable to connect to Qdrant server.")
                sys.exit()
            else:
                await asyncio.sleep(1)
    
    if res["exists"] == False:
        await Db.http(Db.Meth.PUT, "", {
            "vectors": {
                "size": Config.LLAMA.EMBEDDING.SIZE,
                "distance": "Cosine"
            }
        })

        await Db.http(Db.Meth.PUT, "/points?wait=true", {
            "points": [{
                "id": _UUID0,
                "vector": [0.0] * Config.LLAMA.EMBEDDING.SIZE,
                "payload": {
                    "sources": []
                }
            }]
        })

@benchmark("db create")
async def create(input: Iterable[dict], src: str) -> bool:
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

        await Db.http(Db.Meth.PUT, "/points?wait=true", {
            "points": points
        })

        count += len(points)

    return await _update_journal(src, count) # TODO rollback needed if failed?

async def read(vector: list[float]) -> str:
    res = await Db.http(Db.Meth.POST, "/points/search", {
        "vector": vector,
        "with_payload": True,
        "limit": Config.QDRANT.READ_LIMIT,
        "score_threshold": 0.65
    })

    ret = ""
    for hit in res:
        if hit["payload"].get("document") is not None:
            if ret != "":
                ret += "\n"
            ret += hit["payload"]["document"]

    return ret.strip() # in case some document are actually ""

async def list() -> list[any]:
    """
    returns: [{ name, count, timestamp }]
    """
    res = await Db.http(Db.Meth.POST, "/points", {
        "ids": [_UUID0],
        "with_payload": True
    })

    return res[0]["payload"]["sources"]

@benchmark("db delete")
async def delete(src: str) -> bool:
    await Db.http(Db.Meth.POST, "/points/delete?wait=true", {
        "filter": {
            "must": [{
                "key": "source",
                "match": {
                    "value": src
                }
            }]
        }
    })

    return await _update_journal(src, -1) # TODO rollback needed if failed?
    
async def _update_journal(src: str, count: int) -> bool:
    """
    count: > 0, add entry to journal. duplicate src name is allowed\n
    0 or -int, delete ALL entries with src name
    """
    sources = await list()

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

    res = await Db.http(Db.Meth.POST, "/points/payload?wait=true", {
        "payload": {
            "sources": sources
        },
        "points": [_UUID0]
    })

    return res["status"] == "completed"

async def drop(collection: str):
    # use full path as collection name is diff!
    await Db.http(Db.Meth.DEL, f"{Config.QDRANT.HOST}/collections/{collection}", None)
