# set before import of modules that uses config
import config
from config import ConfigKey
config.set(ConfigKey.BENCHMARK, True)

import unittest
import db
import llm

class TestIntegration(unittest.TestCase):
    def test_crud_flow(self):
        collection = "_test_"
        src="ffx"
        query = "where can wakka be found in the beginning?"

        print("\n\ninit..")
        db.drop(collection) # drop just in case prev test did not cleanup properly

        print("\ncreating..")
        c_res = db.create(collection, "./test/test.txt", src, 500)
        self.assertTrue(c_res[0])

        print("\nreading..")
        ctx = db.read(collection, query)
        print("\033[34m" + llm.inference(ctx, query) + "\033[0m")

        print("\ndeleting..")
        self.assertTrue(db.delete(collection, src))

        print("\nread non-existing..")
        ctx = db.read(collection, query)
        ans = llm.inference(ctx, query)
        print("\033[34m" + ans + "\033[0m")
        self.assertTrue(ans == "Unable to answer as no data can be found in the record.")

        print("\ndropping..")
        self.assertTrue(db.drop(collection))
