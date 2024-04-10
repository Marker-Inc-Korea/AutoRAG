# Hybrid - cc

The `hybrid-cc` module is designed to retrieve passages from multiple retrievals. Similar to the `hybrid-rrf` module, the `hybrid-cc` module is also aimed at retrieving information from multiple sources.

However, it distinguishes itself by using the Convex Combination (CC) algorithm. 
This algorithm allows for the calculation of scores with varying weights between each retrieval, offering a flexible approach to combining retrieval scores.

## ❗️Hybird additional explanation

By default, `hybrid` is designed to be used without writing target_module_params. Other modules listed in target_modules
must be included in the config file, and hybrid is calculated based on the best of the results from those modules.

Once evaluated to find the optimal pipeline, extracting the pipeline creates a parameter called target_module_params. This helps the hybrid work even if you don't include other modules, which is useful in test dataset evaluation and deployment situations.

Also, target_modules and target_module_params must be in the form of a tuple. By default, tuples don't work in yaml files, but AutoRAG specifically uses them. In the AutoRAG config yaml file, a tuple is a tuple of parameters, as opposed to a List, which is a list of options for a parameter that you can try for optimization. Note that because we are using `ast.literal_eval()`, we have to write tuples as if we were writing them in python.

So something like `('bm25', 'vectordb')` with quotes will work.

## **Module Parameters**
- **Parameters**: `target_modules`, `weights`, `target_module_params`
- **Purpose**: This module combines different retrieval modules (target_modules) and applies weights to them, adjusting their influence on the final retrieval outcome. The `target_module_params` allows for further customization of each target module.

```{attention}
In the config YAML file that you wrote, you don't have to specify the target_module_params. 
It is automatically generated when you run the optimization process.
```

## **Example config.yaml**
```yaml
modules:
  - module_type: hybrid_cc
    target_modules: ('bm25', 'vectordb')
    weights:
      - (0.5, 0.5)
      - (0.3, 0.7)
      - (0.7, 0.3)
```
