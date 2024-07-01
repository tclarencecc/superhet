# top level warning suppressors
# 'Unable to close http connection', qdrant logs in :root so setting level on root is necessary
import logging
logging.getLogger().setLevel(logging.ERROR)

# set before import of modules (that may use config)!
import config
from config import ConfigKey
config.set(ConfigKey.BENCHMARK, True)

import db
import llm

query = "where can wakka be found in the beginning?"

    

ops = "read"



if ops == "create":
    db.create("my collection", "./xxx.txt", "ffx", 500)
elif ops == "read":
    ctx = db.read("my collection", query)
    print("\n" + llm.inference(ctx, query) + "\n")
elif ops == "delete":
    db.delete("my collection", "ffx")
elif ops == "drop":
    db.drop("my collection")
