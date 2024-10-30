import sqlite3
from typing import Iterable

from app.config import Config
from app.util import Hnsw

class Sql:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Sql, cls).__new__(cls)

        return cls._instance

    def __init__(self):
        self._conn: sqlite3.Connection = None
        self._cursor: sqlite3.Cursor = None

    def start(self):
        self._conn = sqlite3.Connection(Config.STORAGE.SQL)
        self._cursor = self._conn.cursor()

    def stop(self):
        self._conn.close()

    # implement enter & exit as async since Sql "with"-wraps asyncio internals
    async def __aenter__(self):
        self.start()
        
    async def __aexit__(self, type, value, traceback):
        self.stop()

    @staticmethod
    def exec(qs: str, *args: any, lastrowid=False, fetch=False) -> int | list[any]:
        """
        output type:
        lastrowid: return id of the last inserted row
        fetch: return queried rows
        * only 1 type of output can be enabled at a time
        """
        assert Sql._instance is not None

        param = ()
        for arg in args:
            param += (arg,) # trailing comma for () needed to be considered a tuple!

        cursor = Sql._instance._cursor
        cursor.execute(qs, param)

        if lastrowid:
            return cursor.lastrowid
        elif fetch:
            return cursor.fetchall()

    @staticmethod
    def commit():
        assert Sql._instance is not None
        Sql._instance._conn.commit()
        

class Vector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Vector, cls).__new__(cls)

        return cls._instance
    
    def __init__(self):
        self._hnsw = Hnsw("cosine", Config.LLAMA.EMBEDDING.SIZE)

        try:
            self._hnsw.load(Config.STORAGE.INDEX, allow_replace_deleted=True)
        except RuntimeError:
            # index file loading failed, initialize first then save file
            if Config.STORAGE.HNSW.DEBUG:
                print("creating hnsw index")
            
            self._hnsw.init_index(Config.STORAGE.HNSW.RESIZE_STEP,
                M=Config.STORAGE.HNSW.M,
                ef_construction=Config.STORAGE.HNSW.EF_CONSTRUCTION,
                allow_replace_deleted=True
            )
            self._hnsw.ef = Config.STORAGE.HNSW.EF_SEARCH
            self._save_index()

        Sql.exec("CREATE TABLE IF NOT EXISTS vector(document TEXT, source TEXT)")

    def _save_index(self):
        self._hnsw.save(Config.STORAGE.INDEX)
    
    def create(self, input: Iterable[dict], src: str):
        while (dv := next(input, None)) is not None:
            ids = []
            count = dv["len"]

            for i in range(count):
                id = Sql.exec("INSERT INTO vector(document, source) VALUES (?,?)", dv["documents"][i], src,
                    lastrowid=True
                )
                ids.append(id)
            
            if count + self._hnsw.element_count >= self._hnsw.max_elements:
                if Config.STORAGE.HNSW.DEBUG:
                    print(f"resizing hnsw index to {self._hnsw.max_elements + Config.STORAGE.HNSW.RESIZE_STEP}")
                self._hnsw.resize(self._hnsw.max_elements + Config.STORAGE.HNSW.RESIZE_STEP)

            self._hnsw.add(dv["vectors"], ids, replace_deleted=True)

        # rollback-handling:
        # - sql INSERT fails: sql not committed, hnsw unchanged
        # - hnsw add fails: sql not committed, TODO rollback already added vectors?
        self._save_index()
        Sql.commit()
            
    def read(self, vector: list[float]) -> str:
        ret = ""
        id = None
        dist = Config.STORAGE.HNSW.MIN_DISTANCE

        try:
            ids, dists = self._hnsw.query(vector, k=Config.STORAGE.HNSW.K)
        except RuntimeError:
            # hnswlib knn_query raises this when query returns nothing
            return ret

        for i in range(len(ids)):
            if dists[i][0] < dist:
                dist = dists[i][0]
                id = ids[i][0]

        if id is not None:
            res = Sql.exec("SELECT document FROM vector WHERE rowid=?", id, fetch=True)
            if len(res) == 1:
                ret = res[0][0]
            else:
                # corresponding id is in index but not in sql!
                raise SystemError(f"Data and index entry mismatch, row id: {id}")

        return ret

    def delete(self, src: str):
        for id in Sql.exec("SELECT rowid FROM vector WHERE source=?", src, fetch=True): # [(1,) ..]
            self._hnsw.delete(id[0])

        Sql.exec("DELETE FROM vector WHERE source=?", src)

        # rollback-handling:
        # - hnsw delete fails: TODO unmark already deleted vectors? sql unchanged
        # - sql DELETE fails: TODO unmark all deleted vectors, sql not committed
        self._save_index()
        Sql.commit()

    def list(self) -> list[tuple[str, int]]:
        # [(src, count) ..]
        return Sql.exec("SELECT source, COUNT(*) AS count FROM vector GROUP BY source ORDER BY count", fetch=True)
