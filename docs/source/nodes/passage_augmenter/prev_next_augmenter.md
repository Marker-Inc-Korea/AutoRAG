# Prev Next Augmenter

This module is inspired by
LlamaIndex ['Forward/Backward Augmentation'](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/PrevNextPostprocessorDemo/).
It allows users to fetch additional passages.

## **Module Parameters**

- **num_passages** : The number of passages to add before and after the retrieved passage
  Default is 1.
- **mode** : The mode of augmentation
    - `prev`: add passages before the retrieved passage
    - `next`: add passages after the retrieved passage
    - `both`: add passages before and after the retrieved passage

  Default is 'next'.

## **Example config.yaml**

```yaml
modules:
  - module_type: prev_next_augmenter
    num_passages: 1
    mode: next
```
