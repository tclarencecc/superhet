import asyncio
from argparse import ArgumentParser, ArgumentError
import shlex
import sys

from app.chunker import Chunker
import app.db as db
import app.llm as llm
from app.util import new_async_task, PrintColor
from app.config import Config

_CMD_HELP = Config.CLI_CMD_PREFIX + "help"
_CMD_LIST = Config.CLI_CMD_PREFIX + "list"
_CMD_CREATE = Config.CLI_CMD_PREFIX + "create"
_CMD_DELETE = Config.CLI_CMD_PREFIX + "delete"

class _ArgsParserQuery(Exception): ...

class _ArgsParser(ArgumentParser):
    # override due to:
    # https://github.com/python/cpython/issues/103498#issuecomment-1520641460
    def exit(self, status=0, message=None):
        pass # do nothing; just prevent exiting
        
    def error(self, message: str):
        # subparsers can also raise error but only the root's will reach the top
        # HACK if below error msg came from root means user inputted a non-command i.e. a query
        if self.prog == "root" and message.startswith("argument command: invalid choice:"):
            raise _ArgsParserQuery
        else:
            raise ArgumentError(None, message)

async def cli():
    # wait for qdrant to be ready & prepare collection if it does not exists yet
    await db.init()

    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    
    PrintColor.BLUE(f"\nAsk me anything or {_CMD_HELP} for options")

    parser = _ArgsParser(
        prog="root",
        add_help=False,
        usage="""
    {list}                                List all sources
    {create} FILE -s NAME                 Create data from FILE
      -s NAME, --source NAME             Group under this source
    {delete} SOURCE                       Delete all data with SOURCE group""".format(
        list=_CMD_LIST,
        create=_CMD_CREATE,
        delete=_CMD_DELETE
    ))
    
    sub = parser.add_subparsers(dest="command")

    # help
    sub.add_parser(_CMD_HELP)

    # list
    sub.add_parser(_CMD_LIST)

    # create FILE -s SOURCE
    create_parser = sub.add_parser(_CMD_CREATE)
    create_parser.add_argument("file")
    create_parser.add_argument("-s", "--source", type=str, required=True)

    # delete SOURCE
    delete_parser = sub.add_parser(_CMD_DELETE)
    delete_parser.add_argument("source")
    
    lock = False # cli edit lock flag

    def callback():
        nonlocal lock
        lock = False

    def async_task(coro):
        nonlocal lock
        lock = True
        new_async_task(coro, callback)

    while True:
        bytes = await reader.read(500)
        
        # while cli is locked, ignore all user inputs
        if lock == False:
            input = bytes.decode("utf-8").replace("\n", "")

            try:
                arg = parser.parse_args(shlex.split(input))
                if arg.command == _CMD_HELP:
                    parser.print_usage()

                elif arg.command == _CMD_LIST:
                    async def coro_list():
                        list = await db.list()

                        print(f"total {len(list)}")
                        print(f"{'source':<20}  {'rows':>5}  created")
                        for li in list:
                            print(f"{li["name"]:<20}  {li["count"]:>5}  {li["timestamp"]}")

                    async_task(coro_list())
                    
                elif arg.command == _CMD_CREATE:
                    async def coro_create():
                        chunker = Chunker(arg.file, {
                            "size": 256,
                            "overlap": 0.15
                        })
                        embed = llm.Embedding(chunker)
                        await db.create(embed, arg.source)
                    async_task(coro_create())

                elif arg.command == _CMD_DELETE:
                    async def coro_delete():
                        await db.delete(arg.source)
                    async_task(coro_delete())

            except _ArgsParserQuery:
                    async def coro_read():
                        vec = llm.Embedding.create(input)
                        ctx = await db.read(vec)
                        res = llm.completion_stream(ctx, input)

                        for r in res:
                            PrintColor.BLUE(r, stream=True)
                        print("\n")

                    async_task(coro_read())

            except ArgumentError as e:
                print(e.message)
