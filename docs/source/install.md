# Installation and Setup

To install AutoRAG, you can use pip:

```bash
pip install AutoRAG
```

```{admonition} Trouble with installation?
Do you have any trouble with installation?
First, you can check out the [troubleshooting](troubleshooting.md) page.
```

## Setup OPENAI API KEY
To use LLM and embedding models, it is common to use OpenAI models.
If you want to use other models, check out [here](local_model.md)

You need to set OPENAI_API_KEY environment variable.
You can get your API key at [here](https://platform.openai.com/account/api-keys).

```bash
export OPENAI_API_KEY="sk-...your-api-key..."
```

Or, as an alternative, you can set env variable using .env file.

```bash
pip install python-dotenv
```

Then, make .env file at your root folder like below.
```dotenv
OPENAI_API_KEY=sk-...your-api-key...
```

And when you try to run AutoRAG, you can use below code to load .env file.

```python
from dotenv import load_dotenv

load_dotenv()
```

And you are ready to use AutoRAG!


## Build from source

If you want to build AutoRAG from source, the first step is to clone AutoRAG repository.

```bash
git clone https://github.com/Marker-Inc-Korea/AutoRAG.git
```

And install AutoRAG to editable.
```bash
cd AutoRAG
pip install -e .
```

And then, for testing and documentation build, you need to install some additional packages.

```bash
pip install -r tests/requirements.txt
pip install -r docs/requirements.txt
```

For testing, you have to set env variable at pytest.ini.
Make new pytest.ini file at the root of the project and write below.

```ini
[pytest]
env =
    OPENAI_API_KEY=sk-...your-api-key...

log_cli=true
log_cli_level=INFO
```

After that, you can run tests with pytest.

```bash
python -m pytest -n auto
```

After this, please check out our documentation for contributors. 
We are writing this documentation for contributors, so please wait for a while.
