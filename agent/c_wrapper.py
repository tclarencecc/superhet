import hnswlib
import numpy as np

class Hnsw:
    def __init__(self, space: str, dim: int):
        self._hnsw = hnswlib.Index(space, dim)

    @property
    def max_elements(self) -> int:
        return self._hnsw.max_elements
    
    @property
    def element_count(self) -> int:
        return self._hnsw.element_count
    
    @property
    def ef(self) -> int:
        return self._hnsw.ef

    @ef.setter # setter req corresponding getter!
    def ef(self, val: int):
        self._hnsw.set_ef(val)

    def init_index(self, max_elements: int, M=16, ef_construction=200, random_seed=100, allow_replace_deleted=False):
        self._hnsw.init_index(
            max_elements,
            M=M,
            ef_construction=ef_construction,
            random_seed=random_seed,
            allow_replace_deleted=allow_replace_deleted
        )

    def load(self, path: str, max_elements=0, allow_replace_deleted=False):
        self._hnsw.load_index(
            path,
            max_elements=max_elements,
            allow_replace_deleted=allow_replace_deleted
        )

    def save(self, path: str):
        self._hnsw.save_index(path)

    def resize(self, val: int):
        self._hnsw.resize_index(val)

    def add(self, data: list[float], ids: list[int], num_threads=-1, replace_deleted=False):
        self._hnsw.add_items(
            np.array(data),
            np.array(ids),
            num_threads=num_threads,
            replace_deleted=replace_deleted
        )

    def delete(self, id: int):
        self._hnsw.mark_deleted(id)

    def query(self, data: list[float], k=1, num_threads=-1, filter=None) -> tuple[list[int], list[float]]:
        lbls, dists = self._hnsw.knn_query(
            np.array(data),
            k=k,
            num_threads=num_threads,
            filter=filter
        )
        return (lbls.tolist(), dists.tolist())
