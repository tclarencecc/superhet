import asyncio
from argparse import ArgumentParser, ArgumentError
import shlex
import sys
import time

from app.chunker import Chunker
from app.storage import Vector
from app.llm import Embedding, Completion, Chat
from app.util import PrintColor
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
    
    chat = Chat()

    # cli edit lock
    lock_time = 0.0

    while True:
        bytes = await reader.read(500)
        
        # if buffered cmd/s are less than X msec from last cmd, its likely inputted while prev cmd is still running
        if time.time() - lock_time < 0.25:
            continue

        input = bytes.decode("utf-8").replace("\n", "")

        try:
            try:
                spl_in = shlex.split(input)
            except ValueError:
                raise _ArgsParserQuery

            arg = parser.parse_args(spl_in)
            
            # only valid commands can reach this point
            if arg.command == _CMD_HELP:
                parser.print_usage()

            elif arg.command == _CMD_LIST:
                list = Vector.list()

                print(f"total {len(list)}")
                print(f"{'source':<20}  {'rows':>5}")
                for li in list:
                    print(f"{li[0]:<20}  {li[1]:>5}")

            elif arg.command == _CMD_CREATE:
                chunker = Chunker(arg.file)
                embed = Embedding(chunker)
                Vector.create(embed, arg.source)
                print("done")

            elif arg.command == _CMD_DELETE:
                Vector.delete(arg.source)
                print("done")

            lock_time = time.time()

        except _ArgsParserQuery:
                vec = Embedding.from_string(input)
                ctx = Vector.read(vec)
                res = Completion.run(input, ctx, chat)
                
                for r in res:
                    PrintColor.BLUE(r, stream=True)
                print("\n")

                lock_time = time.time()
                
        except ArgumentError as e:
            print(e.message)
