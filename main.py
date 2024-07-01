# top level warning suppressors
# 'Unable to close http connection', qdrant logs in :root so setting level on root is necessary
import logging
logging.getLogger().setLevel(logging.ERROR)

import db
import llm
from util import benchmark

query = "where can wakka be found in the beginning?"

@benchmark("create")
def create():
    db.create("my collection", "./xxx.txt", "ffx", 500)

@benchmark("read")
def read() -> str:
    return db.read("my collection", query)
    
@benchmark("infer")
def infer(ctx: str):
    print(llm.inference(ctx, query) + "\n")

@benchmark("delete")
def delete():
    db.delete("my collection", "ffx")

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
