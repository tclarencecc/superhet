from unittest import IsolatedAsyncioTestCase, skipIf
import asyncio
from httpx import AsyncClient

import app.db as db
from app.llm import Embedding, Completion, Chat
from app.chunker import Chunker
from app.config import Config
import config_test
from app.util import PrintColor

http: AsyncClient = None
database: db.Db = None

class TestIntegration(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        global http
        global database

        http = AsyncClient()
        database = db.Db(http)
        database.start()

        n_embd, n_ctx = Embedding.stats()
        Config.LLAMA.EMBEDDING.SIZE = n_embd
        Config.LLAMA.EMBEDDING.CONTEXT = n_ctx

        Config.STRICT_CTX_ONLY = True

    @classmethod
    def tearDownClass(cls):
        database.stop()

    @skipIf(config_test.SKIP_INT_CRUD, "")
    async def test_crud(self):
        # disable 'Executing Task...took # seconds' warning
        asyncio.get_event_loop().set_debug(False)

        collection = "_test_collection_"
        src="python"
        query = "how does python manage memory?"
        #nrf = "unable to answer"

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
        embed = Embedding(chunker)
        c_res = await db.create(embed, src)
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
            vec = Embedding.create(query)
            ctx = await db.read(vec)

            chat = Chat()
            completion = Completion()
            res = completion(query, ctx, chat)

            for r in res:
                PrintColor.BLUE(r, stream=True)
            print("\n")

            return chat.latest.res

        print("reading..")
        propans = await read()
        self.assertTrue(propans != "")

        print("deleting..")
        self.assertTrue(await db.delete(src))

        # print("read non-existing..")
        # nonex = await read()
        # print(f"nonex = {nonex}")
        # self.assertTrue(nonex == nrf)

        print("dropping..")
        await db.drop(collection)

        await http.aclose()
