import pathlib
from collections.abc import Iterable
from typing import Any

import click
import pandas as pd
from datasets import load_dataset


PASSAGES_URL = (
	"https://huggingface.co/datasets/PrimeQA/clapnq_passages/resolve/main/passages.tsv"
)
RETRIEVAL_URL = "https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval"
TRAIN_QUESTIONS_URL = f"{RETRIEVAL_URL}/train/question_train_answerable.tsv"
TEST_QUESTIONS_URL = f"{RETRIEVAL_URL}/dev/question_dev_answerable.tsv"


def _load_tsv(url: str):
	return load_dataset("csv", data_files={"split": url}, delimiter="\t", split="split")


def _records_to_dataframes(
	passages: Iterable[dict[str, Any]], questions: Iterable[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
	corpus = pd.DataFrame(
		{
			"doc_id": passage["id"],
			"contents": passage["text"],
			"metadata": {"title": passage.get("title", "")},
		}
		for passage in passages
	)
	corpus_ids = set(corpus["doc_id"])
	qa_rows = []
	for question in questions:
		doc_id = question.get("doc-id-list")
		answers = [
			answer.strip() for answer in (question.get("answers") or "").split("::")
		]
		answers = list(dict.fromkeys(answer for answer in answers if answer))
		if not doc_id or doc_id not in corpus_ids or not answers:
			continue
		qa_rows.append(
			{
				"qid": str(question["id"]),
				"query": question["question"].strip(),
				"retrieval_gt": [[doc_id]],
				"generation_gt": answers,
			}
		)

	return corpus, pd.DataFrame(qa_rows)


@click.command()
@click.option(
	"--save_path",
	type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
	default=pathlib.Path(__file__).parent,
	help="Path to save sample ClapNQ dataset.",
)
def load_clapnq_dataset(save_path: pathlib.Path):
	"""Download the official ClapNQ retrieval corpus and QA splits."""
	paths = {
		"corpus": save_path / "corpus.parquet",
		"train": save_path / "qa_train.parquet",
		"test": save_path / "qa_test.parquet",
	}
	if any(path.exists() for path in paths.values()):
		raised = next(path for path in paths.values() if path.exists())
		raise ValueError(f"{raised.name} already exists")

	passages = _load_tsv(PASSAGES_URL)
	train_questions = _load_tsv(TRAIN_QUESTIONS_URL)
	test_questions = _load_tsv(TEST_QUESTIONS_URL)
	corpus, train_qa = _records_to_dataframes(passages, train_questions)
	_, test_qa = _records_to_dataframes(passages, test_questions)

	save_path.mkdir(parents=True, exist_ok=True)
	corpus.to_parquet(paths["corpus"], index=False)
	train_qa.to_parquet(paths["train"], index=False)
	test_qa.to_parquet(paths["test"], index=False)


if __name__ == "__main__":
	load_clapnq_dataset()
