from sample_dataset.clapnq.load_clapnq_dataset import _records_to_dataframes


def test_records_to_dataframes_uses_official_passage_ids_and_answers():
	passages = [
		{"id": "doc-1", "text": "Evidence.", "title": "Title"},
		{"id": "doc-2", "text": "Negative context.", "title": "Other"},
	]
	questions = [
		{
			"id": 1,
			"question": "What is the answer?",
			"doc-id-list": "doc-1",
			"answers": "Answer::Answer",
		},
		{
			"id": 2,
			"question": "Another question",
			"doc-id-list": "doc-2",
			"answers": "Second answer",
		},
	]

	corpus, qa = _records_to_dataframes(passages, questions)

	assert corpus["doc_id"].tolist() == ["doc-1", "doc-2"]
	assert qa["retrieval_gt"].tolist() == [[["doc-1"]], [["doc-2"]]]
	assert qa["generation_gt"].tolist() == [["Answer"], ["Second answer"]]


def test_records_to_dataframes_ignores_questions_without_gold_passages():
	passages = [{"id": "doc-1", "text": "Context", "title": "Title"}]
	questions = [
		{
			"id": "missing-doc",
			"question": "Question",
			"doc-id-list": None,
			"answers": "Answer",
		},
		{
			"id": "unknown-doc",
			"question": "Question",
			"doc-id-list": "doc-2",
			"answers": "Answer",
		},
	]

	corpus, qa = _records_to_dataframes(passages, questions)

	assert len(corpus) == 1
	assert qa.empty
