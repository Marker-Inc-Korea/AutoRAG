import os
import pathlib

from datasets import load_dataset


def load_triviaqa_corpus():
    # set file path
    file_path = "MarkrAI/triviaqa_sample_autorag"

    # load dataset
    corpus_dataset = load_dataset(file_path, "corpus").to_pandas()

    # path setting
    sample_dataset_dir = pathlib.PurePath(__file__).parent.parent
    project_dir = os.path.join(sample_dataset_dir, "triviaqa")

    # save corpus data
    corpus_dataset.to_parquet(os.path.join(project_dir, "corpus.parquet"), index=False)


if __name__ == '__main__':
    load_triviaqa_corpus()
