from unittest import TestCase, skipIf

import config_test
from app.config import PromptFormat
from common.toml import Toml

class TestConfig(TestCase):

    @skipIf(config_test.SKIP_CONFIG, "")
    def test_config(self):
        class TestConfig1:
            class _root:
                REQ = Toml.Spec("root.req")
            ROOT = _root

            class _nested:
                class _child:
                    REQ = Toml.Spec("nested.child.req")
                    CASTED = Toml.Spec("nested.child.casted", None, lambda x: PromptFormat[x])
                    OPTIONAL = Toml.Spec("nested.child.optional", False)
                CHILD = _child
            NESTED = _nested

        with Toml("./test/t1.toml") as t:
            t.load_to(TestConfig1)

        self.assertEqual(TestConfig1.ROOT.REQ, "data1")
        self.assertEqual(TestConfig1.NESTED.CHILD.REQ, "data2")
        self.assertEqual(TestConfig1.NESTED.CHILD.CASTED, PromptFormat.CHATML)
        self.assertEqual(TestConfig1.NESTED.CHILD.OPTIONAL, False)

        # --------------------------------------------------------
        class TestConfig2:
            class _root:
                REQ = Toml.Spec("root.req2")
            ROOT = _root

        errmsg = ""
        with Toml("./test/t1.toml") as t:
            try:
                t.load_to(TestConfig2)
            except Exception as e:
                errmsg = str(e)

        self.assertEqual(errmsg, "Key 'root.req2' not found in toml file.")

        # --------------------------------------------------------
        class TestConfig3:
            class _root:
                REQ = Toml.Spec("malformed.nested.xxx")
            ROOT = _root

        with Toml("./test/t1.toml") as t:
            try:
                t.load_to(TestConfig3)
            except Exception as e:
                errmsg = str(e)

        self.assertEqual(errmsg, "Key 'malformed.nested.xxx' not found in toml file.")
