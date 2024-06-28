from db import Database
from llm import Llm

CLI_PATH = "./llama-cli"
MODEL_PATH = "../llama/models/qwen2-1_5b-instruct-q4_k_m.gguf"
DB_HOST = "http://localhost:6333"
LLM_HOST = "http://127.0.0.1:8080"

db = Database(host=DB_HOST, collection="my collection")


ops = "read"



if ops == "insert":
    db.insert("ffx", file="./input.txt")
elif ops == "read":
    query = "where can wakka be found?"
    ctx = db.read(query)

    llm = Llm(host=LLM_HOST)
    print("\n\n\n" + llm.run(ctx, query) + "\n\n\n")
elif ops == "delete":
    db.delete("ffx")
