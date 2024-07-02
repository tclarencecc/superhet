import unittest
import chunker

TEST_TXT_1 = """ZANARKAND STADIUM
After the FMV, you'll be controlling Tidus. Just go south and talk to the
2 girls or the 3 children. You can rename him if you want. Afterwards the
game will begin and Tidus will leave. When you regain control, just keep
going east. On the next screen head north into the building. You'll watch
the blitzball game, then Sin will attack Zanarkand. When you regain control,
head south towards the screen and you'll meet up with Auron. After their
short talk about Sin, you'll have to fight some sinscales. These are easy,
just attack. After that there will be a boss."""

TEST_TXT_2 = """After their
short talk about Sin, you'll have to fight some sinscales. These are easy,
just attack. After that there will be a boss."""

class TestChunker(unittest.TestCase):
    def test_sliding_window(self):
        ret = chunker.split(TEST_TXT_1, 100)
        self.assertEqual(ret[-1], TEST_TXT_2)
