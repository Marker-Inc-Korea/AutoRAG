from sample_dataset.clapnq.load_clapnq_dataset import _records_to_dataframes


def test_records_to_dataframes_deduplicates_passages_and_preserves_answers():
	records = [
		{
			"id": 1,
			"input": "What is the answer?",
			"passages": [{"title": "Title", "text": " Evidence. "}],
			"output": [{"answer": "Answer"}, {"answer": "Answer"}],
		},
		{
			"id": 2,
			"input": "Another question",
			"passages": [{"title": "Title", "text": "Evidence."}],
			"output": [{"answer": "Second answer"}],
		},
	]

	corpus, qa = _records_to_dataframes(records)

	assert len(corpus) == 1
	assert corpus.iloc[0]["contents"] == "Evidence."
	assert qa["retrieval_gt"].tolist() == [
		[[corpus.iloc[0]["doc_id"]]],
		[[corpus.iloc[0]["doc_id"]]],
	]
	assert qa["generation_gt"].tolist() == [["Answer"], ["Second answer"]]


def test_records_to_dataframes_ignores_unanswerable_and_empty_passages():
	records = [
		{
			"id": "unanswerable",
			"input": "Question",
			"passages": [{"title": "Title", "text": "Context"}],
			"output": [{"answer": ""}],
		},
		{
			"id": "empty",
			"input": "Question",
			"passages": [{"title": "Title", "text": "  "}],
			"output": [{"answer": "Answer"}],
		},
	]

	corpus, qa = _records_to_dataframes(records)

	assert len(corpus) == 1
	assert corpus.iloc[0]["contents"] == "Context"
	assert qa.empty
