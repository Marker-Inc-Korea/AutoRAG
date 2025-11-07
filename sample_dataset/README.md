# sample_dataset handling

The sample_dataset folder does not includes a `qa.parquet`, `corpus.parquet` file that is significantly large and cannot be uploaded directly to Git due to size limitations.

To prepare and use datasets available in the sample_dataset folder, specifically `triviaqa`, `hotpotqa`, `msmarco` and `eli5`, you can follow the outlined methods below.

## Usage

 The example provided uses `triviaqa`, but the same approach applies to `msmarco`, `eli5` and `hotpotqa`.

### 1. Run with a specified save path
To execute the Python script from the terminal and save the dataset to a specified path, use the command:

```bash
python ./sample_dataset/triviaqa/load_triviaqa_dataset.py --save_path /path/to/save/dataset
```
This runs the `load_triviaqa_dataset.py` script located in the `./sample_dataset/triviaqa/` directory,
using the `--save_path` argument to specify the dataset's save location.

### 2. Run without specifying a save path
If you run the script without the `--save_path` argument, the dataset will be saved to a default location, which is the directory containing the `load_triviaqa_dataset.py` file, essentially `./sample_dataset/triviaqa/`:
```bash
python ./sample_dataset/triviaqa/load_triviaqa_dataset.py
```
This behavior allows for a straightforward execution without needing to specify a path, making it convenient for quick tests or when working directly within the target directory.
