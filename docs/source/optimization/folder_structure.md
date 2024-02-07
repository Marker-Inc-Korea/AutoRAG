# Folder Structure

## Sample Structure Index
- [Project](#project)
  - [trial](#trial)
    - [config.yaml](#configyaml)
    - [summary.csv](#trial-summarycsv)
    - [Node_line_name](#pre-retrieve-node-line)
      - [summary.csv](#node-line-summarycsv)
      - [Node name](#query-expansion)
        - 0.parquet
        - ...
        - best_(index).parquet
        - [summary.csv](#node-summarycsv)
    - [Node_line_name](#retrieve-node-line)
      - Node name
        - ...
      - Node name
        - ...
  - [data](#data)
  - [resources](#resources)
  - [trial.json](#trialjson)


## Project
In a project, you experiment with only one dataset.
The project folder is where the user runs from.
    
![project_folders](../_static/project_folders.png)
        
### trial
        
This will contain the results of a single run of the yaml file. If there are multiple of these, it means you ran multiple experiments. Consider running multiple trials on the same data.
        
The folder names are determined by the number of trials run. The first trial folder is named `0`, the second trial folder is named `1`, and so on.
        
![trial_folder](../_static/trial_folder.png)
        
#### config.yaml
The yaml file we used for this experiment
```{Tip}
You can see a sample full [config.yaml](sample_full_config.yaml).
```
#### [trial] summary.csv
Full trial summary csv file


Node lines, selected modules, files and parameters used by the selected modules, and the time it took to process one row.
    
![trail_summary](../_static/trial_summary.png)
    
#### pre_retrieve_node_line
![node_line_folder](../_static/node_line_folder.png)

    
    
##### [Node Line] summary.csv
![node_line_summary](../_static/node_line_summary.png)

Contains the best modules and settings selected from each node.
You can see the node, the selected modules, their files and parameters used, and the time it took to process a row.
        
        
##### query_expansion
Node names belonging to the node_line
        
![node_folder](../_static/node_folder.png)
        
Depending on the module and module params, you can run different experiments on a node. The following photo shows three experiments on a node.
        
- 0.parquet
- 1.parquet
- …
- best_(index).parquet ⇒ Top results on a node

```{tip}
In the picture, the first result is the best of the three experiments, so the file is named best_0
```

  
###### [Node] summary.csv
Results for each node. All attempts and evaluation metric results are recorded.
  
![node_summary](../_static/node_summary.png)
     
#### retrieve_node_line

```{attention}
It is organized in the same format as above. It would be too long to explain it all, so we won't explain it here.
```

    
### data

![data_folder](../_static/data_folder.png)

- corpus.parquet ⇒ corpus dataset
- qa.parquet ⇒ qa dataset
  ```{tip}
  QA data can exist only as qa.parquet, but it is recommended to split it into train and test for more accurate optimization. See the following() for how to build a qa dataset and corpus dataset.
  ```

### resources

![resources_folder](../_static/resources_folder.png)

- `bm25.pkl`: created when using bm25
- `chroma`: created when using vectordb
    - collection_name = the name of the `embedding model`

### trial.json
        
It contains information about each trial.
        
![trial_json](../_static/trial_json.png)
