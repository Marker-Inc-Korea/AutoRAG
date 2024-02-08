# API endpoint

## Running API server

As mentioned in the tutorial, you can run api server from extracted YAML file or trial folder as follows:

```python
from autorag.deploy import Runner

runner = Runner.from_yaml('your/path/to/pipeline.yaml')
runner.run_api_server()

runner = Runner.from_trial_folder('your/path/to/trial_folder')
runner.run_api_server()
```

```bash
autorag run_api --config_path your/path/to/pipeline.yaml --host 0.0.0.0 --port 8000
```

## API Endpoint

You can use AutoRAG api server using `/run` endpoint.
It is a `POST` operation, and you can specify a user query as `query` and result column as `result_column` in the request body.
Then, you can get a response with result looks like `{'result_column': result}` 
The `result_column` is the same as the `result_column` you specified in the request body.
And the `result_column` must be one of the last output of your pipeline. The default is 'answer.' 

```bash
curl -X POST "http://your-host:your-port/run" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"query\":\"your question\", \"result_column\":\"your_result_column\"}"
```

```python
import requests

url = "http://your-host:your-port/run"
payload = "{\"query\":\"your question\", \"result_column\":\"your_result_column\"}"
headers = {
    'accept': "application/json",
    'Content-Type': "application/json"
    }

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)
```
