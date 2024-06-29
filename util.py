import time
import functools

def benchmark(name: str):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*arg, **kwargs):
            t = time.time()
            ret = fn(*arg, **kwargs)
            t = time.time() - t

            tf = "sec"
            if t < 0.1:
                t = t * 1000
                tf = "ms"

            print("{name}: {t:.3f} {tf}\n".format(name=name, t=t, tf=tf))
            return ret
        return wrapper
    return decorate
