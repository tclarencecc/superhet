import uvicorn

from relay.config import Config

Config.load()

if Config.IN_PROD:
    # ssl paths are default certbot folder locations
    uvicorn.run("relay.main:app", host="0.0.0.0", port=443,
        ssl_keyfile="/etc/letsencrypt/live/superhet.top/privkey.pem",
        ssl_certfile="/etc/letsencrypt/live/superhet.top/cert.pem",
        ssl_ca_certs="/etc/letsencrypt/live/superhet.top/chain.pem"
    )
else:
    uvicorn.run("relay.main:app", host="0.0.0.0", port=80)
