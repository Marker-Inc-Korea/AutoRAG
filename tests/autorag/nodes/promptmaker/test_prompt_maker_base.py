import pandas as pd

prompt = "Answer this question: {query} \n\n {retrieved_contents}"
queries = ["What is the capital of Japan?", "What is the capital of China?"]
retrieved_contents = [
	[
		"Tokyo is the capital of Japan.",
		"Tokyo, the capital of Japan, is a huge metropolitan city.",
	],
	[
		"Beijing is the capital of China.",
		"Beijing, the capital of China, is a huge metropolitan city.",
	],
]
retrieve_scores = [[0.9, 0.8], [0.9, 0.8]]
retrieved_ids = [["doc1", "doc2"], ["doc3", "doc4"]]
previous_result = pd.DataFrame(
	{
		"query": queries,
		"retrieved_contents": retrieved_contents,
		"retrieve_scores": retrieve_scores,
		"retrieved_ids": retrieved_ids,
	}
)

doc_id = ["doc1", "doc2", "doc3", "doc4", "doc5"]
contents = [
	"This is a test document 1.",
	"This is a test document 2.",
	"This is a test document 3.",
	"This is a test document 4.",
	"This is a test document 5.",
]
metadata = [
	{"window": "havertz arsenal doosan minji naeun gaeun lets go"} for _ in range(5)
]
corpus_df = pd.DataFrame({"doc_id": doc_id, "contents": contents, "metadata": metadata})

retrieved_metadata = [
	[
		{"window": "havertz arsenal doosan minji naeun gaeun lets go"},
		{"window": "havertz arsenal doosan minji naeun gaeun lets go"},
	],
	[
		{"window": "havertz arsenal doosan minji naeun gaeun lets go"},
		{"window": "havertz arsenal doosan minji naeun gaeun lets go"},
	],
]
