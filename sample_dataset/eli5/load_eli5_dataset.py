import os
import pathlib

from datasets import load_dataset


def load_eli5_dataset():
    # set file path
    file_path = "MarkrAI/eli5_sample_autorag"

    # load dataset
    corpus_dataset = load_dataset(file_path, "corpus")['train'].to_pandas()
    qa_dataset = load_dataset(file_path, "qa")['train'].to_pandas()

    # path setting
    sample_dataset_dir = pathlib.PurePath(__file__).parent.parent
    project_dir = os.path.join(sample_dataset_dir, "eli5")

    # save data
    if os.path.exists(os.path.join(project_dir, "corpus.parquet")) is True:
        raise ValueError("corpus.parquet already exists")
    if os.path.exists(os.path.join(project_dir, "qa.parquet")) is True:
        raise ValueError("qa.parquet already exists")
    corpus_dataset.to_parquet(os.path.join(project_dir, "corpus.parquet"), index=False)
    qa_dataset.to_parquet(os.path.join(project_dir, "qa.parquet"), index=False)


if __name__ == '__main__':
    load_eli5_dataset()
