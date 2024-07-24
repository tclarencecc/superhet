--------
run test
- run in project root folder (not in /test)

python3 -m unittest discover test -v

--------------------------
remove unused dependencies
- remove all package in req.txt; update req.txt afterwards
- re-install currently used packages (add more here as needed); update req.txt again

pip3 uninstall -r requirements.txt -y
pip3 freeze > requirements.txt
pip3 install 'qdrant-client[fastembed]'
pip3 install aiohttp
pip3 install pyyaml
pip3 install pyinstaller
pip3 freeze > requirements.txt

---------
build exe

pyinstaller main.spec
