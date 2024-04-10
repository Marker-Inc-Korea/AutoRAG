# Hybrid - rrf

The `hybrid_rrf` module is designed to retrieve passages from multiple retrievals. 
The `hybrid_rrf` module is tailored for retrieving passages from multiple sources of information. It utilizes the Reciprocal Rank Fusion (RRF) algorithm to calculate final similarity scores. This calculation is based on the ranking of passages in each retrieval, effectively combining retrieval scores from different sources.

## ❗ Hybird additional explanation

By default, `hybrid` is designed to be used without writing target_module_params. Other modules listed in target_modules
must be included in the config YAML file, and hybrid is calculated based on the best of the results from those modules.

Once evaluated to find the optimal pipeline, extracting the pipeline creates a parameter called target_module_params. This helps the hybrid work even if you don't include other modules, which is useful in test dataset evaluation and deployment situations.

Also, target_modules and target_module_params must be in the form of a tuple. By default, tuples don't work in yaml files, but AutoRAG specifically uses them. In the AutoRAG config YAML file, a tuple is a tuple of parameters, as opposed to a List, which is a list of options for a parameter that you can try for optimization. Note that because we are using `ast.literal_eval()`, we have to write tuples as if we were writing them in python.

So something like `('bm25', 'vectordb')` with quotes will work.

## **Module Parameters**
- **Parameters**: `target_modules`, `rrf_k`, `target_module_params`
- **Functionality**: Configures the Hybrid RRF module by specifying target modules, the RRF constant 'k'.
`rrf_k` parameter can take a huge impact of the performance.

```{attention}
In the config YAML file that you wrote, you don't have to specify the target_module_params. 
It is automatically generated when you run the optimization process.
```

## **Example config.yaml**
```yaml
modules:
  - module_type: hybrid_rrf
    target_modules: ('bm25', 'vectordb')
    rrf_k: [3, 5, 10]
```
