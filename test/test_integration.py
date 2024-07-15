from unittest import IsolatedAsyncioTestCase, skipIf
import subprocess
import shlex
import signal
import warnings
import db
import llm
from chunker import Chunker
import asyncio
import config_test
from config import Config

_proc_db = None
_proc_llm = None

class TestIntegration(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        global _proc_db
        global _proc_llm

        _proc_db = subprocess.Popen(
            shlex.split(Config.QDRANT.SHELL),
            cwd=Config.QDRANT.PATH,
            env=Config.QDRANT.ENV,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        _proc_llm = subprocess.Popen(
            shlex.split(Config.LLAMA.SHELL),
            cwd=Config.LLAMA.PATH,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    @classmethod
    def tearDownClass(cls):
        _proc_db.send_signal(signal.SIGINT)
        _proc_llm.send_signal(signal.SIGINT)

        print("pid " + str(_proc_db.pid) + " retcode " + str(_proc_db.wait()))
        print("pid " + str(_proc_llm.pid) + " retcode " + str(_proc_llm.wait()))

    @skipIf(config_test.SKIP_INT_CRUD, "")
    async def test_crud(self):
        # disable 'Executing Task...took # seconds' warning
        asyncio.get_event_loop().set_debug(False)

        # block 'Api key is used with an insecure connection'
        warnings.filterwarnings("ignore", module="qdrant_client")

        # sleep to let db & llm complete init
        await asyncio.sleep(4)

        collection = "_test_"
        src="ffx"
        query = "how does python manage memory?"

        print("\n\ninit..")
        await db.drop(collection) # drop just in case prev test did not cleanup properly

        print("\ncreating..")
        chunker = Chunker("./test/t1.txt", {
            "size": 250,
            "overlap": 0.15
        })
        c_res = await db.create(collection, chunker, src)
        self.assertTrue(c_res > 0)

        async def read() -> str:
            ctx = await db.read(collection, query)
            ans = await llm.completion(ctx, query)
            print(ans)
            return ans

        print("\nreading..")
        await read()

        print("\ndeleting..")
        self.assertTrue(await db.delete(collection, src))

        print("\nread non-existing..")
        nonex = await read()
        self.assertTrue(nonex == "Unable to answer as no data can be found in the record.")

        print("\ndropping..")
        self.assertTrue(await db.drop(collection))
