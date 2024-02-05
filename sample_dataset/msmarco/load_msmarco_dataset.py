import os
import pathlib
import click

from datasets import load_dataset


@click.command()
@click.option('--save_path', type=str, default=pathlib.PurePath(__file__).parent, help='Path to save sample msmarco dataset.')
def load_msmarco_dataset(save_path):
    # set file path
    file_path = "MarkrAI/msmarco_sample_autorag"

    # load dataset
    corpus_dataset = load_dataset(file_path, "corpus")['train'].to_pandas()
    qa_train_dataset = load_dataset(file_path, "qa")['train'].to_pandas()
    qa_test_dataset = load_dataset(file_path, "qa")['test'].to_pandas()

    # save corpus data
    if os.path.exists(os.path.join(save_path, "corpus.parquet")) is True:
        raise ValueError("corpus.parquet already exists")
    if os.path.exists(os.path.join(save_path, "qa.parquet")) is True:
        raise ValueError("qa.parquet already exists")
    corpus_dataset.to_parquet(os.path.join(save_path, "corpus.parquet"), index=False)
    qa_train_dataset.to_parquet(os.path.join(save_path, "qa_train.parquet"), index=False)
    qa_test_dataset.to_parquet(os.path.join(save_path, "qa_test.parquet"), index=False)


if __name__ == '__main__':
    load_msmarco_dataset()
