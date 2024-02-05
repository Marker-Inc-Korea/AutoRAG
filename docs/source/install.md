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

And you are ready to use AutoRAG!
