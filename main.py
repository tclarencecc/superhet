# top level warning suppressors
# 'Unable to close http connection', qdrant logs in :root so setting level on root is necessary
import logging
logging.getLogger().setLevel(logging.ERROR)

from db import Database
from llm import Llm
from util import benchmark

DB_HOST = "http://localhost:6333"
LLM_HOST = "http://127.0.0.1:8080"

query = "where can wakka be found in the beginning?"

db = Database(host=DB_HOST, collection="my collection")

@benchmark("create")
def create():
    db.create("./xxx.txt", "ffx", 500)

@benchmark("read")
def read() -> str:
    return db.read(query)
    
@benchmark("infer")
def infer(ctx: str):
    llm = Llm(host=LLM_HOST)
    print(llm.infer(ctx, query) + "\n")

@benchmark("delete")
def delete():
    db.delete("ffx")

@benchmark("drop")
def drop():
    db.drop("my collection")



ops = "read"



if ops == "create":
    create()
elif ops == "read":
    ctx = read()
    infer(ctx)
elif ops == "delete":
    delete()
elif ops == "drop":
    drop()
