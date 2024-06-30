from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import uuid
import chunker
import os

# qdrant fastembed reads from this env-var for embedding model path
os.environ["FASTEMBED_CACHE_PATH"] = "./fastembed"

class Database:
    def __init__(self, host="", collection="") -> None:
        if host == "":
            self.client = QdrantClient(":memory:")
            self.collection = "temp" # db is temp only, just use any name for collection
        else:
            self.client = QdrantClient(host)
            if collection == "":
                raise Exception("Database server mode has undefined collection.")
            self.collection = collection
    
    def create(self, input: str, src: str, chunk: int, alphabet=True) -> None:
        documents = list[str]

        if input.startswith("./"):
            try:
                f = open(input)
            except FileNotFoundError:
                print(input + " not found.")
            else:
                documents = chunker.split(f.read(), chunk, alphabet=alphabet)
        elif input.startswith("<!DOCTYPE html>"):
            # TODO
            pass
        else:
            documents = chunker.split(input, chunk, alphabet=alphabet)

        ids = []
        metadata = []
        for doc in documents:
            ids.append(uuid.uuid4().hex)
            metadata.append({ "source": src })

        self.client.add(self.collection, documents, metadata=metadata, ids=ids)

        # indexing payload.source may be necessary..
        # https://qdrant.tech/documentation/concepts/indexing/

    def read(self, query: str, limit=1) -> str:
        if query == "":
            raise Exception("Database.read undefined query.")

        hits = self.client.query(self.collection, query, limit=limit)
        ret = ""
        
        for hit in hits:
            if ret != "":
                ret = ret + "\n"
            # score in hit.metadata["score"], if needed
            ret = ret + hit.metadata["document"]
        return ret
        
    def delete(self, src: str) -> None:
        self.client.delete(self.collection, points_selector=Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=src)
            )]
        ))

    def drop(self, collection: str) -> None:
        self.client.delete_collection(collection)
    