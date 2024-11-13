import tomllib
import inspect
from typing import Callable

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
