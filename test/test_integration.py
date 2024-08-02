from unittest import IsolatedAsyncioTestCase, skipIf
import subprocess
import shlex
import signal
import warnings
import asyncio

import app.db as db
import app.llm as llm
from app.chunker import Chunker
from app.config import Config
import config_test

_proc_db = None

class TestIntegration(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        global _proc_db
        _proc_db = subprocess.Popen(
            shlex.split(Config.QDRANT.SHELL),
            cwd=Config.QDRANT.PATH,
            env=Config.QDRANT.ENV,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    @classmethod
    def tearDownClass(cls):
        _proc_db.send_signal(signal.SIGINT)
        print("\npid " + str(_proc_db.pid) + " retcode " + str(_proc_db.wait()))

    @skipIf(config_test.SKIP_INT_CRUD, "")
    async def test_crud(self):
        # disable 'Executing Task...took # seconds' warning
        asyncio.get_event_loop().set_debug(False)

        # block 'Api key is used with an insecure connection'
        warnings.filterwarnings("ignore", module="qdrant_client")

        # sleep to let db & llm complete init
        await asyncio.sleep(4)

        collection = "_test_collection_"
        src="python"
        query = "how does python manage memory?"
        nrf = "Unable to answer as no data can be found in the record."

        # override default collection and start with a blank collection
        # repeated create/delete on same collection often results to 
        # inconsistent search behavior!
        print("\ninit..")
        Config.COLLECTION = collection
        await db.init()

        print("creating..")
        chunker = Chunker("./test/t1.txt", {
            "size": 250,
            "overlap": 0.25
        })
        c_res = await db.create(chunker, src)
        self.assertTrue(c_res)

        print("listing..")
        list = await db.list()
        inlist = False
        for li in list:
            if li["name"] == src:
                self.assertTrue(li["count"] == 7)
                inlist = True
        
        self.assertTrue(inlist)

        async def read() -> str:
            ctx = await db.read(query)
            ans = llm.completion(ctx, query)
            print(ans)
            return ans

        print("reading..")
        propans = await read()
        self.assertTrue(propans != nrf)

        print("deleting..")
        self.assertTrue(await db.delete(src))

        print("read non-existing..")
        nonex = await read()
        self.assertTrue(nonex == nrf)

        print("dropping..")
        await db.drop(collection)
