from unittest import IsolatedAsyncioTestCase, skipIf
import asyncio
import os

from app.storage import Sql, Vector
from app.llm import Embedding, Completion, Chat
from app.chunker import Chunker
from app.config import Config
import config_test
from app.util import PrintColor

sql: Sql = None

def cleanup():
    if os.path.isfile(Config.STORAGE.INDEX):
        os.remove(Config.STORAGE.INDEX)

    if os.path.isfile(Config.STORAGE.SQL):
        os.remove(Config.STORAGE.SQL)

class TestIntegration(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        global sql

        def post_config_load():
            n_embd, n_ctx = Embedding.stats()
            Config.LLAMA.EMBEDDING.SIZE = n_embd
            Config.LLAMA.EMBEDDING.CONTEXT = n_ctx

        Config.load_from_toml(post_config_load)

        Config.STRICT_CTX_ONLY = True
        Config.STORAGE.SQL = "../data/test_data"
        Config.STORAGE.INDEX = "../data/test_index"
        
        cleanup()

        sql = Sql()
        sql.start()

        _ = Completion._llm()

    @classmethod
    def tearDownClass(cls):
        sql.stop()

    @skipIf(config_test.SKIP_INT_CRUD, "")
    def test_crud(self):
        # disable 'Executing Task...took # seconds' warning
        asyncio.get_event_loop().set_debug(False)

        src="python"
        query = "what is python"

        print("creating..")
        chunker = Chunker("./test/t1.txt")
        embed = Embedding(chunker)
        Vector().create(embed, src)

        print("listing..")
        list = Vector().list()
        inlist = False
        for li in list:
            if li[0] == src:
                self.assertTrue(li[1] == 7)
                inlist = True
        
        self.assertTrue(inlist)

        def read():
            vec = Embedding.from_string(query)
            ctx = Vector().read(vec)

            chat = Chat()
            res = Completion.run(query, ctx, chat)

            for r in res:
                PrintColor.BLUE(r, stream=True)
            print("\n")

            return chat.latest.answer

        print("reading..")
        ans = read()
        self.assertTrue(type(ans) is str)

        print("deleting..")
        Vector().delete(src)

        print("read non-existing..")
        ans = read()
        self.assertTrue(ans is None)

        cleanup()
