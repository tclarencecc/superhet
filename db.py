from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance, Filter, FieldCondition, MatchValue
import uuid
import chunker
import warnings

# suppress `resume_download` deprecated warning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

_stm = SentenceTransformer("all-MiniLM-L6-v2")

class Database:
    def __init__(self, host="", collection="") -> None:
        if host == "":
            self.client = QdrantClient(":memory:")
            self.collection = "temp" # db is temp only, just use any name for collection
        else:
            self.client = QdrantClient(host)
            if collection == "":
                raise Exception("Database server mode have undefined collection.")
            self.collection = collection
    
    def create(self, input: str, src: str, chunk: int, alphabet=True) -> None:
        sentences = list[str]

        if input.startswith("./"):
            try:
                f = open(input)
            except FileNotFoundError:
                print(input + " not found.")
            else:
                sentences = chunker.split(f.read(), chunk, alphabet=alphabet)
        elif input.startswith("<!DOCTYPE html>"):
            # TODO
            pass
        else:
            sentences = chunker.split(input, chunk, alphabet=alphabet)

        if self.client.collection_exists(self.collection) == False:
            self.client.create_collection(self.collection, VectorParams(
                size=_stm.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ))
        self.client.upload_points(
            self.collection,
            [PointStruct(
                id=uuid.uuid4().hex,
                vector=_stm.encode(sentence).tolist(),
                payload={ "text": sentence, "source": src }
            )
            for sentence in sentences],
        )

        # indexing payload.source may be necessary..
        # https://qdrant.tech/documentation/concepts/indexing/

    def read(self, query: str, limit=1, combine=True) -> str | list[tuple[str, float]]:
        if query == "":
            raise Exception("Database.read undefined query.")

        hits = self.client.search(
            self.collection,
            query_vector=_stm.encode(query).tolist(),
            limit=limit,
        )

        if combine:
            ret = ""
            for hit in hits:
                ret = ret + "\n\n" + hit.payload["text"]
            return ret
        else:
            ret = []
            for hit in hits:
                ret.append((hit.payload["text"], hit.score))
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
    