# sample_dataset handling

The sample_dataset folder does not includes a `qa.parquet`, `corpus.parquet` file that is significantly large and cannot be uploaded directly to Git due to size limitations. 

To manage this, we have developed Python script named `load_dataset.py` that can efficiently load this dataset file and put it in each project folder.
