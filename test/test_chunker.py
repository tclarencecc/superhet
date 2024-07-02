import unittest
import chunker
from chunker import Chunker

class TestChunker(unittest.TestCase):
    def test_sliding_window(self):
        # de-indented
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

        ret = chunker.split(s1, 100)
        self.assertEqual(ret[-1], s2)

    def test_chunker_read(self):
        c = []
        for chunk in Chunker("./test/t1.txt"):
            c.append(chunk)

        s1 = """Python uses dynamic typing and a combination of reference counting and a cycle-detecting 
garbage collector for memory management. It uses dynamic name resolution (late binding), 
which binds method and variable names during program execution."""

        self.assertEqual(len(c), 8)
        self.assertEqual(c[1], s1)

        c.clear()
        for chunk in Chunker("./test/t2.txt", separator="<br>"):
            c.append(chunk)

        self.assertEqual(len(c), 7)
        self.assertEqual(c[0], "abc<br")
        self.assertEqual(c[1], "def<br")
        self.assertEqual(c[2], "ghi<brx")
        self.assertEqual(c[3], "pqr<br<br")
        self.assertEqual(c[4], "jkl\n\nmno")
        self.assertEqual(c[5], "")
        self.assertEqual(c[6], "stu\n< b r >")
