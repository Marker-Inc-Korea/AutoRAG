import hashlib
import pathlib
from collections.abc import Iterable
from typing import Any

import click
import pandas as pd
from datasets import load_dataset


DATASET_ID = "PrimeQA/clapnq"
TRAIN_FILE = "clapnq_train_answerable.jsonl"
TEST_FILE = "clapnq_dev_answerable.jsonl"


def _passage_id(title: str, text: str) -> str:
	value = f"{title}\n{text}".encode("utf-8")
	return f"clapnq-{hashlib.sha256(value).hexdigest()[:16]}"


def _records_to_dataframes(
	records: Iterable[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
	corpus: dict[str, dict[str, Any]] = {}
	qa_rows: list[dict[str, Any]] = []

	for record in records:
		passage_ids = []
		for passage in record["passages"]:
			title = passage.get("title", "")
			text = passage["text"].strip()
			if not text:
				continue
			doc_id = _passage_id(title, text)
			corpus.setdefault(
				doc_id,
				{
					"doc_id": doc_id,
					"contents": text,
					"metadata": {"title": title},
				},
			)
			passage_ids.append(doc_id)

		answers = [
			output["answer"].strip()
			for output in record.get("output", [])
			if output.get("answer", "").strip()
		]
		answers = list(dict.fromkeys(answers))
		if passage_ids and answers:
			qa_rows.append(
				{
					"qid": str(record["id"]),
					"query": record["input"].strip(),
					"retrieval_gt": [passage_ids],
					"generation_gt": answers,
				}
			)

	return pd.DataFrame(corpus.values()), pd.DataFrame(qa_rows)


def _load_split(filename: str):
	return load_dataset(DATASET_ID, data_files={"split": filename}, split="split")


@click.command()
@click.option(
	"--save_path",
	type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
	default=pathlib.Path(__file__).parent,
	help="Path to save sample ClapNQ dataset.",
)
def load_clapnq_dataset(save_path: pathlib.Path):
	"""Download ClapNQ and write AutoRAG-compatible parquet files."""
	paths = {
		"corpus": save_path / "corpus.parquet",
		"train": save_path / "qa_train.parquet",
		"test": save_path / "qa_test.parquet",
	}
	if any(path.exists() for path in paths.values()):
		raised = next(path for path in paths.values() if path.exists())
		raise ValueError(f"{raised.name} already exists")

	train_corpus, train_qa = _records_to_dataframes(_load_split(TRAIN_FILE))
	test_corpus, test_qa = _records_to_dataframes(_load_split(TEST_FILE))
	corpus = pd.concat([train_corpus, test_corpus]).drop_duplicates("doc_id")

	save_path.mkdir(parents=True, exist_ok=True)
	corpus.to_parquet(paths["corpus"], index=False)
	train_qa.to_parquet(paths["train"], index=False)
	test_qa.to_parquet(paths["test"], index=False)


if __name__ == "__main__":
	load_clapnq_dataset()
