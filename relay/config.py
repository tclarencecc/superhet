import sys
from argparse import ArgumentParser, BooleanOptionalAction

class Config:
    # argv derived
    API_KEY = None
    DEBUG = False

    class _header:
        # hardcoded
        NAME = "Agent-Name"
        KEY = "Agent-Apikey"
    HEADER = _header


    @staticmethod
    def load():
        parser = ArgumentParser()
        parser.add_argument("exe") # ./uvicorn itself, just ignore
        parser.add_argument("--relay-apikey", type=str, required=False)
        parser.add_argument("--relay-debug", action=BooleanOptionalAction, required=False)
        arg = parser.parse_args(sys.argv)

        if arg.relay_apikey is not None:
            Config.API_KEY = arg.relay_apikey

        if arg.relay_debug is not None:
            Config.DEBUG = arg.relay_debug
