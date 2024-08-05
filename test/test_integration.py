from unittest import IsolatedAsyncioTestCase, skipIf
import subprocess
import shlex
import signal
import asyncio
from httpx import AsyncClient

import app.db as db
import app.llm as llm
from app.chunker import Chunker
from app.config import Config
import config_test

proc_db = None
http: AsyncClient = None

class TestIntegration(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        global proc_db
        global http

        proc_db = subprocess.Popen(
            shlex.split(Config.QDRANT.SHELL),
            cwd=Config.QDRANT.PATH,
            env=Config.QDRANT.ENV,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        http = AsyncClient()

    @classmethod
    def tearDownClass(cls):
        proc_db.send_signal(signal.SIGINT)
        retcode = proc_db.wait()
        print(f"\npid {proc_db.pid} retcode {retcode}")

    @skipIf(config_test.SKIP_INT_CRUD, "")
    async def test_crud(self):
        # disable 'Executing Task...took # seconds' warning
        asyncio.get_event_loop().set_debug(False)

        # sleep to let db init
        await asyncio.sleep(3)

        collection = "_test_collection_"
        src="python"
        query = "how does python manage memory?"
        nrf = "Unable to answer as no data can be found in the record."

        # override default collection and start with a blank collection
        # repeated create/delete on same collection often results to 
        # inconsistent search behavior!
        print("\ninit..")
        Config.COLLECTION = collection
        await db.init(http)

        print("creating..")
        chunker = Chunker("./test/t1.txt", {
            "size": 250,
            "overlap": 0.25
        })
        embed = llm.Embedding(chunker)
        c_res = await db.create(embed, src, http)
        self.assertTrue(c_res)

        print("listing..")
        list = await db.list(http)
        inlist = False
        for li in list:
            if li["name"] == src:
                self.assertTrue(li["count"] == 7)
                inlist = True
        
        self.assertTrue(inlist)

        async def read() -> str:
            vec = llm.Embedding.create(query)
            ctx = await db.read(vec, http)
            ans = llm.completion(ctx, query)
            print(ans)
            return ans

        print("reading..")
        propans = await read()
        self.assertTrue(propans != nrf)

        print("deleting..")
        self.assertTrue(await db.delete(src, http))

        print("read non-existing..")
        nonex = await read()
        self.assertTrue(nonex == nrf)

        print("dropping..")
        await db.drop(collection, http)

        await http.aclose()
