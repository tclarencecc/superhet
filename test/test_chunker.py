from unittest import TestCase, skipIf

from app.chunker import Chunker, FileStream, _sliding_window
import config_test
import app.llm as llm
from app.config import Config

class TestChunker(TestCase):
    @classmethod
    def setUpClass(cls):
        def post_config_load():
            n_embd, n_ctx = llm.Embedding.stats()
            Config.LLAMA.EMBEDDING.SIZE = n_embd
            Config.LLAMA.EMBEDDING.CONTEXT = n_ctx

        Config.load_from_toml(post_config_load)

        # override toml; test cases below uses these values
        Config.CHUNK.SIZE = 100
        Config.CHUNK.OVERLAP = 0.25

# -----------------------------------------------------------------------------
    @skipIf(config_test.SKIP_CHUNKER, "")
    def test_sliding_window(self):
        s1 = """ZANARKAND STADIUM
After the FMV, you'll be controlling Tidus. Just go south and talk to the
2 girls or the 3 children. You can rename him if you want. Afterwards the
game will begin and Tidus will leave. When you regain control, just keep
going east. On the next screen head north into the building. You'll watch
the blitzball game, then Sin will attack Zanarkand. When you regain control,
head south towards the screen and you'll meet up with Auron. After their
short talk about Sin, you'll have to fight some sinscales. These are easy,
just attack. After that there will be a boss."""

        s2 = """After their
short talk about Sin, you'll have to fight some sinscales. These are easy,
just attack. After that there will be a boss."""

        ret = _sliding_window(s1)
        self.assertEqual(ret[-1], s2)

# -----------------------------------------------------------------------------
    @skipIf(config_test.SKIP_CHUNKER, "")
    def test_file_stream(self):
        # test \n\n separator
        c = []
        for chunk in FileStream("./test/t1.txt"):
            c.append(chunk)

        s1 = """Python uses dynamic typing and a combination of reference counting and a cycle-detecting 
garbage collector for memory management. It uses dynamic name resolution (late binding), 
which binds method and variable names during program execution."""

        self.assertEqual(len(c), 8)
        self.assertEqual(c[1], s1)

        # test <br> separator (& its mangled variants)
        Config.CHUNK.SEPARATOR = "<br>"

        c.clear()
        for chunk in FileStream("./test/t2.txt"):
            c.append(chunk)

        self.assertEqual(len(c), 7)
        self.assertEqual(c[0], "abc<br")
        self.assertEqual(c[1], "def<br")
        self.assertEqual(c[2], "ghi<brx")
        self.assertEqual(c[3], "pqr<br<br")
        self.assertEqual(c[4], "jkl\n\nmno")
        self.assertEqual(c[5], "")
        self.assertEqual(c[6], "stu\n< b r >")

# -----------------------------------------------------------------------------
    @skipIf(config_test.SKIP_CHUNKER, "")
    def test_chunker(self):
        # test filestream -> sliding_window -> chunker pipeline
        c = []
        for sentence in Chunker("./test/t3.txt"):
            c.append(sentence)

        s1 = """With Tidus,
open the Trigger Command menu again then choose Pincer Attack. They will now
attack Tros from both sides, and this time he can't escape so just pummel him
until he dies."""

        self.assertEqual(len(c), 20)
        self.assertEqual(c[14], s1)

        # test handling of 'x.x' words
        c.clear()
        for sentence in Chunker("./test/t1.txt"):
            c.append(sentence)

        s2 = """You may also find version numbers with a “+” suffix, e.g. “2.2+”. These are unreleased versions, 
built directly from the CPython development repository. In practice, after a final minor release is made, the 
version is incremented to the next minor version, which becomes the “a0” version, e.g. “2.4a0”. See the Developer’s Guide for more information about the development cycle, and PEP 387 to learn more about 
Python’s backward compatibility policy. See also the documentation for sys.version, sys.hexversion, and sys.version_info."""

        self.assertEqual(c[7], s2)
