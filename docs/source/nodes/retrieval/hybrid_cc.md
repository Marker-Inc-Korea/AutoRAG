---
myst:
  html_meta:
    title: AutoRAG - hybrid cc
    description: Learn about hybrid cc module in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,retrieval,hybrid cc
---
# Hybrid - cc

The `hybrid-cc` module is designed to retrieve passages from multiple retrievals. Similar to the `hybrid-rrf` module, the `hybrid-cc` module is also aimed at retrieving information from multiple sources.

However, it distinguishes itself by using the Convex Combination (CC) algorithm.
This algorithm allows for the calculation of scores with varying weights between each retrieval, offering a flexible approach to combining retrieval scores.

## ❗️Hybrid additional explanation

By default, `hybrid` is designed to be used without writing target_module_params. Other modules listed in target_modules
must be included in the config file, and hybrid is calculated based on the best of the results from those modules.

Once evaluated to find the optimal pipeline, extracting the pipeline creates a parameter called target_module_params. This helps the hybrid work even if you don't include other modules, which is useful in test dataset evaluation and deployment situations.

```{attention}
You don't have to specify the module that you want to fuse. It will auto-detect the best module name and parameter for each lexcial and semantic modules.
```

`## **Node Parameters**

- (Required) **top_k**: Essential parameter for retrieval node.
  `
## **Module Parameters**

- (Required) **normalize_method**: The normalization method to use.
  There is some normalization method that you can use at the hybrid cc method.
  AutoRAG support following.
    - `mm`: Min-max scaling
    - `tmm`: Theoretical min-max scaling
    - `z`: z-score normalization
    - `dbsf`: 3-sigma normalization
- (Optional) **weight_range**: The range of the weight that you want to explore. If the weight is 1.0, it means the
  weight to the semantic module will be 1.0 and weight to the lexical module will be 0.0.
  You have to input this value as tuple. It looks like this. `(0.2, 0.8)`. Default is `(0.0, 1.0)`.
- (Optional) **test_weight_size**: The size of the weight that is tested for optimization.
If the weight range is `(0.2, 0.8)` and the size is 7, it will evaluate the following weights.
`0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8`. Default is 101.
- (Optional) **semantic_theoretical_min_value**: This value used by `tmm` normalization method. You can set the
  theoretical minimum value by yourself. Default is -1.
- (Optional) **lexical_theoretical_min_value**: This value used by `tmm` normalization method. You can set the
  theoretical minimum value by yourself. Default is 0.

## **Example config.yaml**
```yaml
modules:
  - module_type: hybrid_cc
    normalize_method: [ mm, tmm, z, dbsf ]
    weight_range: (0.0, 1.0)
    test_weight_size: 101
    lexical_theoretical_min_value: 0
    semantic_theoretical_min_value: -1
```
