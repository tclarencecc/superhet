import uvicorn

uvicorn.run("relay.main:app", host="localhost", port=8765)
