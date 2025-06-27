def cast_retrieve_infos(previous_result):
	if "retrieved_contents" in previous_result.columns:
		return {
			"retrieved_contents": previous_result["retrieved_contents"].tolist(),
			"retrieved_ids": previous_result["retrieved_ids"].tolist(),
			"retrieve_scores": previous_result["retrieve_scores"].tolist(),
		}
	elif "retrieved_contents_semantic" in previous_result.columns:
		return {
			"retrieved_contents": previous_result[
				"retrieved_contents_semantic"
			].tolist(),
			"retrieved_ids": previous_result["retrieved_ids_semantic"].tolist(),
			"retrieve_scores": previous_result["retrieve_scores_semantic"].tolist(),
		}
	elif "retrieved_contents_lexical" in previous_result.columns:
		return {
			"retrieved_contents": previous_result[
				"retrieved_contents_lexical"
			].tolist(),
			"retrieved_ids": previous_result["retrieved_ids_lexical"].tolist(),
			"retrieve_scores": previous_result["retrieve_scores_lexical"].tolist(),
		}
	else:
		raise ValueError(
			"previous_result must contain either 'retrieved_contents', 'retrieved_contents_semantic', or 'retrieved_contents_lexical' columns."
		)
