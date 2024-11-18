import time
from datetime import datetime, timezone

class PrintColor:
    @staticmethod
    def OK(input: str, stream=False):
        PrintColor._print("\033[92m", input, stream)

    @staticmethod
    def WARN(input: str, stream=False):
        PrintColor._print("\033[93m", input, stream)

    @staticmethod
    def ERROR(input: str, stream=False):
        PrintColor._print("\033[91m", input, stream)

    @staticmethod
    def BLUE(input: str, stream=False):
        PrintColor._print("\033[94m", input, stream)

    @staticmethod
    def CYAN(input: str, stream=False):
        PrintColor._print("\033[96m", input, stream)

    @staticmethod
    def _print(color: str, input: str, stream: bool):
        if stream:
            print(f"{color}{input}\033[0m", end="", flush=True)
        else:
            print(f"{color}{input}\033[0m")

def print_duration(name: str, t: float):
    t = time.time() - t
    if t >= 1:
        # 1.1 sec
        PrintColor.OK(f"{name}: {t:.1f} sec")
    elif t < 1 and t >= 0.1:
        # 0.11 sec
        PrintColor.OK(f"{name}: {t:.2f} sec")
    elif t < 0.1 and t >= 0.001:
        # 99 ms
        t = t * 1000
        PrintColor.OK(f"{name}: {t:.0f} ms")
    else: # < 0.001
        # 999 μs
        t = t * 1000000
        PrintColor.OK(f"{name}: {t:.0f} μs")

def timestamp() -> str:
    return datetime.now(
        datetime.now(timezone.utc).astimezone().tzinfo
    ).replace(microsecond=0).isoformat()
