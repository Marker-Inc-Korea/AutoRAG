from typing import List

import pandas as pd


dont_know_phrases = {
	"en": [
		"I don't know",
		"I do not know",
		"Don't know",
		"Do not know",
	],
	"ko": [
		"몰라요",
		"모르겠습니다",
		"모르겠어요",
		"몰라",
		"내가 어떻게 알아?",
		"모르겠소",
		"몰라유",
		"모르것는디",
		"모르겠어유",
		"모르겠네유",
		"모르겠네요",
	],
}


def dontknow_filter_rule_based(qa_df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
	assert (
		"generation_gt" in qa_df.columns
	), "generation_gt column is not in the DataFrame."

	def is_i_dont_know(input_list: List[str], dont_know_phrase: List[str]) -> bool:
		return any(phrase in s for phrase in dont_know_phrase for s in input_list)

	result_df = qa_df[
		~qa_df["generation_gt"].apply(
			lambda x: is_i_dont_know(x, dont_know_phrases[lang])
		)
	]
	return result_df
