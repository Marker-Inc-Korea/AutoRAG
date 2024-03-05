# Web Interface

## Running the Web Interface
As mentioned in the tutorial, you can run the web interface following the below command:

### 1. Use yaml path
```bash
autorag run_web --yaml_path your/path/to/pipeline.yaml
```

```{admonition} Want to specify project folder?
You can specify project directory with `--project_dir` option or project_dir parameter.
```
```bash
autorag run_web --yaml_path your/path/to/pipeline.yaml --project_dir your/project/directory
```

### 2. Use trial path
```bash
autorag run_web --trial_path your/path/to/trial
```

### Web Interface example

You can use the web interface to interact with the AutoRAG pipeline.

The web interface provides a user-friendly environment to input queries and receive responses from the pipeline. The web interface is a convenient way to test the pipeline and observe its performance in real-time.


![Web Interface](../_static/web_interface.png)