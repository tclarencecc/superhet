import asyncio
from argparse import ArgumentParser, ArgumentError
import shlex
import sys
from chunker import Chunker
import db
import llm
from util import new_async_task, PrintColor

async def _coro_create(file: str, collection: str, src: str):
    chunker = Chunker(file, {
        "size": 250, # 250 best so far
        "overlap": 0.25 # 0.25 best so far
    })
    await db.create(collection, chunker, src)

async def _coro_delete(collection: str, src: str):
    await db.delete(collection, src)

async def _coro_read(query: str, collection: str):
    ctx = await db.read(collection, query, 3) # 3 best so far
    ans = await llm.completion(ctx, query)
    PrintColor.BLUE("\n" + ans + "\n")


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
    await asyncio.sleep(5) # let db & llm init first before starting cli

    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    
    PrintColor.BLUE("\nAsk me anything or $help for options\n")

    parser = _ArgsParser(
        prog="root",
        add_help=False,
        usage="""
  $collection NAME                      Collection used for this session
  
  $create FILE -s NAME [-c NAME]        Create data from FILE
    -s NAME, --source NAME              Categorize data under this source
    -c NAME, --collection NAME          Optional. Used instead of NAME in $collection

  $delete SOURCE [-c NAME]              Delete all data with SOURCE category
    -c NAME, --collection NAME          Optional. Used instead of NAME in $collection"""
    )
    
    sub = parser.add_subparsers(dest="command")

    # $help
    sub.add_parser("$help")

    # $collection <name>
    collection_parser = sub.add_parser("$collection")
    collection_parser.add_argument("name")

    # $create <file> -s <source> [-c <collection>]
    create_parser = sub.add_parser("$create")
    create_parser.add_argument("file")
    create_parser.add_argument("-s", "--source", type=str, required=True)
    create_parser.add_argument("-c", "--collection", type=str)

    # $delete <source> [-c <collection>]
    delete_parser = sub.add_parser("$delete")
    delete_parser.add_argument("source")
    delete_parser.add_argument("-c", "--collection", type=str)
    
    lock = False # cli edit lock flag
    collection = "" # collection for this session

    def _callback():
        nonlocal lock
        lock = False

    while True:
        bytes = await reader.read(500)
        
        # while cli is locked, ignore all user inputs
        if lock == False:
            input = bytes.decode("utf-8").replace("\n", "")

            try:
                arg = parser.parse_args(shlex.split(input))
                if arg.command == "$help":
                    parser.print_usage()

                elif arg.command == "$collection":
                    collection = arg.name
                    PrintColor.OK("Collection set to '" + collection + "'")

                elif arg.command == "$create" or arg.command == "$delete":
                    coll = arg.collection
                    if coll == "" or coll is None:
                        coll = collection

                    if coll == "":
                        PrintColor.ERROR("No collection specified. Input with -c option or use $collection command")
                    else:
                        lock = True
                        if arg.command == "$create":
                            new_async_task(_coro_create(arg.file, coll, arg.source), _callback)
                        else:
                            new_async_task(_coro_delete(coll, arg.source), _callback)

            except _ArgsParserQuery:
                if collection == "":
                    PrintColor.ERROR("No collection specified. Input using $collection command")
                else:
                    lock = True
                    new_async_task(_coro_read(input, collection), _callback)

            except ArgumentError as e:
                PrintColor.ERROR(e.message)
