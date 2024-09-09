import pandas as pd
from autorag.data.beta.filter.dontknow import dontknow_filter_rule_based
from autorag.data.beta.schema import QA


def test_dontknow_filter_rule_based():
	en_qa_df = pd.DataFrame(
		{
			"generation_gt": [
				["I don't know what to say", "This is a test"],
				["This is another test", "I do not know the answer"],
				["This is fine", "All good"],
			]
		}
	)

	ko_qa_df = pd.DataFrame(
		{
			"generation_gt": [
				["몰라요", "테스트입니다"],
				["모르겠습니다", "이것은 테스트입니다"],
				["모르겠어요", "이것은 또 다른 테스트입니다"],
				["이것은 괜찮습니다", "모든 것이 좋습니다"],
			]
		}
	)

	# Expected data after filtering
	expected_df_en = pd.DataFrame({"generation_gt": [["This is fine", "All good"]]})

	expected_df_ko = pd.DataFrame(
		{"generation_gt": [["이것은 괜찮습니다", "모든 것이 좋습니다"]]}
	)

	# Test for English
	en_qa = QA(en_qa_df)
	result_en_qa = en_qa.filter(dontknow_filter_rule_based, lang="en").map(
		lambda df: df.reset_index(drop=True)
	)
	pd.testing.assert_frame_equal(result_en_qa.data, expected_df_en)

	# Test for Korean
	ko_qa = QA(ko_qa_df)
	result_ko_qa = ko_qa.filter(dontknow_filter_rule_based, lang="ko").map(
		lambda df: df.reset_index(drop=True)
	)
	pd.testing.assert_frame_equal(result_ko_qa.data, expected_df_ko)
