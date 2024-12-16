# AWS Bedrock x AutoRAG

## Setting Up the AWS profile

First, you need to have an AWS account.

And you need to set up the AWS CLI with your AWS account.

You can find detailed information about the AWS CLI configuration at the [following link](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html)

## Using AWS Bedrock with AutoRAG

For using AWS Bedrock, you can use Llama Index LLm's `bedrock` at the AutoRAG config YAML file without any further configuration.

### Writing the Config YAML File

Here’s the modified YAML configuration using `Bedrock`:

```yaml
nodes:
  - node_line_name: node_line_1
    nodes:
      - node_type: generator
        modules:
          - module_type: llama_index_llm
            llm: bedrock
            model: amazon.titan-text-express-v1
            profile_name: your_profile_name  # Plz replace this with your profile name
```

You can find the model ID at the [following link](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)

multiple models can be used in the same way.

```yaml
nodes:
  - node_line_name: node_line_1
    nodes:
      - node_type: generator
        modules:
          - module_type: llama_index_llm
            llm: bedrock
            model: [amazon.titan-text-express-v1, Claude 3.5 Sonnet, Llama 3.2 90B Instruct]
            profile_name: your_profile_name  # Plz replace this with your profile name
```

For full YAML files, please see the sample_config folder in the AutoRAG repo at [here](https://github.com/Marker-Inc-Korea/AutoRAG/tree/main/sample_config/rag).

### Running AutoRAG

Before running AutoRAG, make sure you have your QA dataset and corpus dataset ready.
If you want to know how to make it, visit [here](../../data_creation/tutorial.md).

Run AutoRAG with the following command:

```bash
autorag evaluate \
 - qa_data_path ./path/to/qa.parquet \
 - corpus_data_path ./path/to/corpus.parquet \
 - project_dir ./path/to/project_dir \
 - config ./path/to/bedrock_config.yaml
```

AutoRAG will automatically experiment and optimize RAG.
