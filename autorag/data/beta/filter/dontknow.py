from typing import Dict

import pandas as pd
from pydantic import BaseModel

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


def dontknow_filter_rule_based(row: Dict, lang: str = "en") -> bool:
	assert (
		"generation_gt" in row.keys()
	), "generation_gt column is not in the DataFrame."
	dont_know_phrase = dont_know_phrases[lang]
	return not any(
		phrase in s for phrase in dont_know_phrase for s in row["generation_gt"]
	)


class Response(BaseModel):
	is_dont_know: bool


def dontknow_filter_openai(qa_df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
	assert (
		"generation_gt" in qa_df.columns
	), "generation_gt column is not in the DataFrame."
