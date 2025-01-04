# vLLM API

To save the reinitialization time, it is great to use vLLM API instead of the original vLLM integration.
You can use openAI like API server, but you can use vLLM API server as well to get a full feature of vLLM.

## Start the vLLM API server

In your vLLM installed machine, start vLLM API server like below.

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ -q awq --port 8012
```

You can find the detail about vLLM API server at the [vLLM documentation](https://docs.vllm.ai/en/stable/getting_started/quickstart.html#openai-compatible-server).

## **Module Parameters**

- **llm**: You can type your 'model name' at here. For example, `facebook/opt-125m`
  or `mistralai/Mistral-7B-Instruct-v0.2`.
- **uri**: The URI of the vLLM API server.
- **max_tokens**: The maximum number of tokens. Default is 4096. Consider using longer tokens for longer prompts.
- **temperature**: The temperature of the sampling. Higher temperature means more randomness.
And support all parameters from vLLM API.

## **Example config.yaml**

```yaml
 - module_type: vllm_api
   uri: http://localhost:8012
   llm: Qwen/Qwen2.5-14B-Instruct-AWQ
   temperature: [0, 0.5]
   max_tokens: 400
```
