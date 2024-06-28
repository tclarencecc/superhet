from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance, Filter, FieldCondition, MatchValue
import uuid
import chunker

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
    
    def insert(self, src: str, file="", txt="", html = "") -> None:
        sens = list[str]
        if file != "":
            try:
                f = open(file)
            except FileNotFoundError:
                print(f + " not found.")
            else:
                sens = chunker.split(f.read(), chunker.SplitType.PARAGRAPH)
        elif txt != "":
            sens = chunker.split(txt, chunker.SplitType.PARAGRAPH)
        elif html != "":
            # TODO
            pass

        if self.client.collection_exists(self.collection) == False:
            self.client.create_collection(self.collection, VectorParams(
                size=_stm.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ))
        self.client.upload_points(
            self.collection,
            [PointStruct(
                id=uuid.uuid4().hex,
                vector=_stm.encode(sen).tolist(),
                payload={ "text": sen, "source": src }
            )
            for sen in sens],
        )

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
    