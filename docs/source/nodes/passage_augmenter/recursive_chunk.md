# Recursive Chunk

This module is inspired by
LlamaIndex ['Recursive Retriever + Node References + Braintrust'](https://docs.llamaindex.ai/en/stable/examples/retrievers/recurisve_retriever_nodes_braintrust/#chunk-references-smaller-child-chunks-referring-to-bigger-parent-chunk).

It allows users to add passages by recursively chunking the retrieved passage.

## **Module Parameters**

- **sub_chunk_sizes** : The list of chunk sizes to split the retrieved passage
    - Default is [128, 256, 512].
- **chunk_overlap** : The overlap between the chunks
    - Default is 20.

## **Example config.yaml**

```yaml
modules:
  - module_type: recursive_chunk
    sub_chunk_sizes: [ [ 128, 256, 512 ], [ 128, 256 ] ]
    chunk_overlap: 20
```
