import pandas as pd
from autorag.data.beta.filter.dontknow import dontknow_filter_rule_based


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
	result_df_en = dontknow_filter_rule_based(en_qa_df, lang="en")
	pd.testing.assert_frame_equal(result_df_en.reset_index(drop=True), expected_df_en)

	# Test for Korean
	result_df_ko = dontknow_filter_rule_based(ko_qa_df, lang="ko")
	pd.testing.assert_frame_equal(result_df_ko.reset_index(drop=True), expected_df_ko)
