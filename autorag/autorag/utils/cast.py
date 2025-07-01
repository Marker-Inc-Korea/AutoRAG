def cast_retrieve_infos(previous_result):
	return {
		"retrieved_contents": cast_retrieved_contents(previous_result),
		"retrieved_ids": cast_retrieved_ids(previous_result),
		"retrieve_scores": cast_retrieve_scores(previous_result),
	}


def cast_retrieved_contents(previous_result):
	if "retrieved_contents" in previous_result.columns:
		return previous_result["retrieved_contents"].tolist()
	elif "retrieved_contents_semantic" in previous_result.columns:
		return previous_result["retrieved_contents_semantic"].tolist()
	elif "retrieved_contents_lexical" in previous_result.columns:
		return previous_result["retrieved_contents_lexical"].tolist()
	else:
		raise ValueError(
			"previous_result must contain either 'retrieved_contents', 'retrieved_contents_semantic', or 'retrieved_contents_lexical' columns."
		)


def cast_retrieved_ids(previous_result):
	if "retrieved_ids" in previous_result.columns:
		return previous_result["retrieved_ids"].tolist()
	elif "retrieved_ids_semantic" in previous_result.columns:
		return previous_result["retrieved_ids_semantic"].tolist()
	elif "retrieved_ids_lexical" in previous_result.columns:
		return previous_result["retrieved_ids_lexical"].tolist()
	else:
		raise ValueError(
			"previous_result must contain either 'retrieved_ids', 'retrieved_ids_semantic', or 'retrieved_ids_lexical' columns."
		)


def cast_retrieve_scores(previous_result):
	if "retrieve_scores" in previous_result.columns:
		return previous_result["retrieve_scores"].tolist()
	elif "retrieve_scores_semantic" in previous_result.columns:
		return previous_result["retrieve_scores_semantic"].tolist()
	elif "retrieve_scores_lexical" in previous_result.columns:
		return previous_result["retrieve_scores_lexical"].tolist()
	else:
		raise ValueError(
			"previous_result must contain either 'retrieve_scores', 'retrieve_scores_semantic', or 'retrieve_scores_lexical' columns."
		)
