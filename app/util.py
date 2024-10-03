import time
import asyncio
from enum import Enum
from datetime import datetime, timezone
from typing import Callable
import tomllib
import inspect

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

def new_async_task(coro, callback: Callable=None):
    def _callback(task: asyncio.Task):
        if callback is not None:
            callback()

        try:
            # needed to avoid 'Task exception was never retrieved'
            # either returns None, exception or raise CancelledError, InvalidStateError
            ex = task.exception()
            if ex is not None:
                raise ex
            
        except Exception as e:
            print(e)

    task = asyncio.get_event_loop().create_task(coro)
    task.add_done_callback(_callback)

class EnumDict:
    def __init__(self):
        self._dict = {}

    def get(self, k: Enum) -> any:
        if self._dict.get(k.name) is None:
            self.set(k)
        
        return self._dict[k.name]

    def set(self, k: Enum, v: any=None):
        if v is not None:
            self._dict[k.name] = v
        else:
            if k.value is None:
                raise ValueError(f"EnumDict enum member '{k.name}' has no preset value. Set a user-defined value first.")

            self._dict[k.name] = k.value

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
        
def timestamp() -> str:
    return datetime.now(
        datetime.now(timezone.utc).astimezone().tzinfo
    ).replace(microsecond=0).isoformat()

class MutableString:
    def __init__(self):
        self._buffer = [""] * 100
        self._idx = -1

    def add(self, input: str):
        if (len(self) + len(input)) > len(self._buffer):
            # extend by input len though this might be more than whats needed
            self._buffer.extend([""] * len(input))

        for char in input:
            self._idx += 1
            self._buffer[self._idx] = char

    def clear(self):
        self._idx = -1

    def value(self):
        # array slice end idx IS excluded
        return "".join(self._buffer[:self._idx + 1])
    
    def not_empty(self) -> bool:
        return self._idx > -1
    
    def strip(self):
        txt = self.value().strip()
        if len(txt) != len(self):
            self.clear()
            self.add(txt)

    def split_len(self, char: str) -> int:
        if len(char) != 1:
            raise ValueError(f"MutableString.split_len only accepts character splitter. Splitter used '{char}'.")
        
        ret = 1 # count current unsplitted text as already 1 partition
        for i in range(len(self)):
            if self._buffer[i] == char:
                ret += 1
        return ret

    def __len__(self) -> int:
        return self._idx + 1

class Toml:
    class Spec:
        def __init__(self, key: str, default: any=None, callback: Callable=None):
            self.key = key
            self.default = default
            self.callback = callback

    def __init__(self, path: str):
        self._path = path
        self._file = None
        self._root: dict[str, any] = None
                
    def __enter__(self):
        try:
            self._file = open(self._path, "rb")
        except:
            raise IOError(f"Unable to open {self._path} toml file.")
        
        self._root = tomllib.load(self._file)
        return self
        
    def __exit__(self, type, value, traceback):
        self._file.close()

    def load_to(self, obj: any):
        subs = list()

        attrs = dir(obj)
        for attr in attrs:
            if attr.startswith("_"):
                continue

            sub = getattr(obj, attr)
            if inspect.isclass(sub):
                subs.append(sub)
            else:
                if type(sub) is Toml.Spec:
                    val = self.parse(sub.key, sub.default)
                    if sub.callback is not None:
                        val = sub.callback(val)

                    setattr(obj, attr, val)

        for sub in subs:
            self.load_to(sub)
            
    def parse(self, key: str, default: any=None) -> any:
        try:
            obj = self._root
            found = True

            for k in key.split("."):
                if obj.get(k) is not None:
                    obj = obj[k]
                else:
                    found = False
                    break

            if found:
                return obj
            
            # if default has value, key is optional
            # if default is none, key is required
            if default is not None:
                return default
            else:
                raise ValueError(f"Key '{key}' not found in toml file.")
        except tomllib.TOMLDecodeError:
            raise ValueError("Error decoding toml file.")
