import asyncio
from argparse import ArgumentParser, ArgumentError
import shlex
import sys
from chunker import Chunker
import db
import llm
from util import new_async_task, PrintColor

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
    # llm should not really take more than 1-2 sec to load, but just in case
    await asyncio.sleep(1)
    while (await llm.ready()) == False:
        await asyncio.sleep(1)

    # in case no collection exists yet
    await db.init()

    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    
    PrintColor.BLUE("\nAsk me anything or $help for options")

    parser = _ArgsParser(
        prog="root",
        add_help=False,
        usage="""
  $list                                 List all sources
  $create FILE -s NAME                  Create data from FILE
    -s NAME, --source NAME              Group under this source
  $delete SOURCE                        Delete all data with SOURCE group"""
    )
    
    sub = parser.add_subparsers(dest="command")

    # $help
    sub.add_parser("$help")

    # $list
    sub.add_parser("$list")

    # $create FILE -s SOURCE
    create_parser = sub.add_parser("$create")
    create_parser.add_argument("file")
    create_parser.add_argument("-s", "--source", type=str, required=True)

    # $delete SOURCE
    delete_parser = sub.add_parser("$delete")
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
                if arg.command == "$help":
                    parser.print_usage()

                elif arg.command == "$list":
                    async def coro_list():
                        list = await db.list()

                        print("total " + str(len(list)))
                        print("{:<20}  {:>5}  {}".format("source", "rows", "created"))
                        for li in list:
                            print("{:<20}  {:>5}  {}".format(
                                li["name"],
                                li["count"],
                                li["timestamp"]
                            ))
                    async_task(coro_list())
                    
                elif arg.command == "$create":
                    async def coro_create():
                        chunker = Chunker(arg.file, {
                            "size": 250, # 250 best so far
                            "overlap": 0.25 # 0.25 best so far
                        })
                        await db.create(chunker, arg.source)
                    async_task(coro_create())

                elif arg.command == "$delete":
                    async def coro_delete():
                        await db.delete(arg.source)
                    async_task(coro_delete())

            except _ArgsParserQuery:
                    async def coro_read():
                        ctx = await db.read(input, 3) # 3 best so far
                        ans = await llm.completion(ctx, input)
                        PrintColor.BLUE(ans + "\n")
                    async_task(coro_read())

            except ArgumentError as e:
                print(e.message)
