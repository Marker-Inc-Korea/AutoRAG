# UPR

The `UPR` module is a reranker based on paper called "[Improving Passage Retrieval with Zero-shot Question Generation](https://arxiv.org/abs/2204.07496)". It uses a language model to generate a question based on the passage and reranks the passages by the likelihood of the question. It can enhance accuracy because it calculates likelihood original passages and related passages (generated) with user's question. 

## **Module Parameters**

Configure the UPR module with parameters like 
- (Optional)`shard_size` (integer): 
  - The larger the shard size, the faster the reranking speed.
          But it will consume more memory and compute power.
  - Default is `16`.
- (Optional) `use_bf16` (boolean):
  - Whether to use bfloat16 for the model. 
  - Default is `False`.
- (Optional) `prefix_prompt` (strings):
  - The prefix prompt serves as the initial context or instruction for the language model.
        It sets the stage for what is expected in the output 
  - Default is `Passage: `
- (Optional) `suffix_prompt` (strings):
  - The suffix prompt provides a cue or a closing instruction to the language model,
              signaling how to conclude the generated text or what format to follow at the end.
  - Default is `Please write a question based on this passage.`

for customizing the reranking behavior.

## **Example config.yaml**
```yaml
modules:
  - module_type: upr
    shard_size: 8
    use_bf16: False
```
