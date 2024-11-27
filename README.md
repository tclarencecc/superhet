# Superhet

Superhet is a local & privacy-first [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) app that lets you *chat with your data* offline or through the web.
It is comprised of 2 components, an **Agent** app running on your local computer and a **Relay** hosted on an internet accessible server.

### Agent
- Self contained CLI-based RAG app
- Built-in vector database and LLM inferencing
- Fully functioning offline
- Run on your own computer or a server behind NAT; Relay makes it internet accessible

### Relay
- Lightweight server app that can accomodate multiple connecting Agents
- Each connected Agents have their own url
- Acts as a bridge to securely expose RAG functionality of connected Agents to the internet

## Install
If [pip-tools](https://pypi.org/project/pip-tools/) is installed, simply run:
```
pip-sync requirements.txt dev-requirements.txt
```

As Superhet uses llama-cpp as its inference engine, depending on your GPU, you may need to include a build flag to enable it.

Refer to [llama-cpp-python supported backends](https://github.com/abetlen/llama-cpp-python#supported-backends) for your device's flag.

For example, if building for Apple Silicon, you will need to include the *Metal* build flag:
```
CMAKE_ARGS="-DGGML_METAL=on" pip-sync requirements.txt dev-requirements.txt
```

## Agent
### Build
Once the packages are installed, build the Agent CLI app by running:
```
pyinstaller agent.spec
```
The executable should now be in the **/dist** folder.


### Configure
Open the config.toml file in the generated folder. The important settings are:
```
[llm]
[llm.completion]
# path relative from executable. must be of "gguf" type
model = "./<model>.gguf"

# refer to model card for prompting format used by model
# valid values: CHATML, GEMMA, LLAMA
prompt_format = "<prompt format>"
```
Download your preferred completion model (GGUF format) and specify its path in the *model* field. Take note that the model should have 
the following prompt format to be supported:
* ChatML
* Gemma
* Llama

```
[llm.embedding]
# path relative from executable. must be of "gguf" type and for embedding use
model = "./<model>.gguf"
```
Download your preferred embedding model (GGUF format) and specify its path in the *model* field.

For english language embedding, [bge-small-en](https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf) is a good choice.

```
[document]
# boundary between paragraphs and chapters
separator = "<separator>"

# document's written script. valid values: LATIN, HANZI
script = "<script>"

[document.chunk]
# number of words per chunk. for latin alphabet documents, is at half of embedding model's token limit
size = <#>

# percent of words that overlap between chunks. between 0 (no overlap) to 0.25
overlap = <#>
```
Only plain text files are supported (for now) as input to the RAG database.

* For *separator*, use **\n\n** if your text file is formatted such that paragraphs and chapters are separated by carriage-return whitespaces.

* For *script*, use **LATIN** if your text file has an alphabetical script.

* For *size*, refer to your embedding model on its token limit and set the size value to half of it. If you used the suggested *bge-small-en* embedding model, 
the size should be **256**.

* For *overlap*, it refers to the percentage of text that is common between each record in the database. Ideal value is **0.15-0.25**

```
[relay]
# address of relay server
host = "<host>"

# api key needed to authenticate with relay server
api_key = "<api key>"

# unique name to identify your AI in the relay server
agent_name = "<agent_name>"
```
* For *host*, set the url of the Relay server (protocol http/s:// can be included)
* For *api_key*, set the secret api key as used in the Relay server
* For *agent_name*, it should be a unique name for the current Relay server you are connecting to


### Run
```
./agent
```
You can now chat by typing in your question.

Superhet is configured out of the box to enable *freeform* chatting (not restricted to your data only).
This restriction can be made configurable in later builds.

View the available commands using
```
!help
```
Insert data into the RAG database using:
```
!create FILE -s SOURCE
```
where FILE is the path relative to the Agent executable and SOURCE is the name of the document

## Relay
### Deploy
For now, there is no docker image to simplify deployment so manually copying over of folders is required.
Copy the following folders to your preferred path:
```
./common
./relay
```

### Dependencies
Depending on your server environment, you may have to activate a Python virtual enviroment before installing these dependencies
```
pip install starlette
pip install uvicorn
pip install websockets
```

### Run
To run the relay, you can either run it via:
* uvicorn directly serving http/https
* same as above but invoke uvicorn inside a python script. Refer to ###./relay.py###
* uvicorn behind nginx

Either way, make sure to include the following required flag when running uvicorn
```
--relay-apikey <YOUR_API_KEY>
```
Also include this optional flag for more verbose logging
```
--relay-debug
```

### Access
Access your Agent from the web
```
<http/s://your_host>/a/<your agent name>
```

## Limitations
### Agent
Agent is only built and tested on Apple Silicon with Metal GPU enabled. Linux builds should *work* 
with the only difference being the build step having CUDA or other flags used instead.

### Relay
Relay only runs as a single instance server (for now). If you run multiple Relays behind a load balancer,
there is no *common bus* to coordinate between instances regarding which Agents and web users are connected
to which machine. 

## License
MIT
