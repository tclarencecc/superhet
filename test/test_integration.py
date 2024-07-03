# set before import of modules that uses config
import config
from config import ConfigKey
config.set(ConfigKey.BENCHMARK, True)

from unittest import IsolatedAsyncioTestCase, skipIf
import db
import llm
import asyncio
import config_test

class TestIntegration(IsolatedAsyncioTestCase):
    @skipIf(config_test.SKIP_INT_CRUD_FLOW, "")
    async def test_crud_flow(self):
        # disable 'Executing Task...took # seconds' warning
        asyncio.get_event_loop().set_debug(False)

        collection = "_test_"
        src="ffx"
        query = "how does python manage memory?"

        print("\n\ninit..")
        await db.drop(collection) # drop just in case prev test did not cleanup properly

        print("\ncreating..")
        c_res = await db.create(collection, "./test/t1.txt", src, 500)
        self.assertTrue(c_res[0])

        async def read() -> str:
            ctx = await db.read(collection, query)
            ans = await llm.inference(ctx, query)
            print("\033[34m" + ans + "\033[0m")
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
