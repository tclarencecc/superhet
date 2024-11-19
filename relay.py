import uvicorn

uvicorn.run("relay.main:app", host="0.0.0.0", port=8765)
