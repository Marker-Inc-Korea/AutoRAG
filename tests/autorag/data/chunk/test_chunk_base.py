import itertools
import os
import pathlib

import pandas as pd

from autorag.data.chunk.base import make_metadata_list

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resource_dir = os.path.join(root_dir, "resources")
data_dir = os.path.join(resource_dir, "chunk_data")

base_texts = [
	"The Kia Tigers lost the Korean Series this year and failed to win the championship. jeffrey went to "
	"gwangju to the Korean Series, but they lost there. I love this story.",
	"minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's BOBB. minsing's Man United beat "
	"estdside_gunn's Chelsea. estdside_gunn always loses. I love this story.",
]
base_metadata = [
	{"path": "kia/lose/jeffrey_tigers_sad_story.pdf"},
	{"path": "chelsea/lose/gunn_chelsea_sad_story.pdf"},
]

parsed_result = pd.read_parquet(os.path.join(data_dir, "sample_parsed.parquet"))

character_expect_texts = list(
	itertools.chain.from_iterable(list(map(lambda x: x.split(". "), base_texts)))
)
expect_texts = {
	"token": [
		"The Kia Tigers lost the Korean Series this year and failed to win the "
		"championship. jeffrey went to gwangju to the Korean Series, but they lost",
		"there. I love this story.",
		"minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's BOBB. "
		"minsing's Man United beat estdside_gunn's Chelsea.",
		"estdside_gunn always loses. I love this story.",
	],
	"token_ko": [
		"파일 제목: jeffrey_tigers_sad_story.pdf\n"
		" 내용: The Kia Tigers lost the Korean Series this year and failed to win the "
		"championship. jeffrey went to gwangju to the Korean Series, but they lost",
		"파일 제목: jeffrey_tigers_sad_story.pdf\n 내용: there. I love this story.",
		"파일 제목: gunn_chelsea_sad_story.pdf\n"
		" 내용: minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's BOBB. "
		"minsing's Man United beat estdside_gunn's Chelsea.",
		"파일 제목: gunn_chelsea_sad_story.pdf\n"
		" 내용: estdside_gunn always loses. I love this story.",
	],
	"token_eng": [
		"file_name: jeffrey_tigers_sad_story.pdf\n"
		" contents: The Kia Tigers lost the Korean Series this year and failed to win "
		"the championship. jeffrey went to gwangju to the Korean Series, but they "
		"lost",
		"file_name: jeffrey_tigers_sad_story.pdf\n"
		" contents: there. I love this story.",
		"file_name: gunn_chelsea_sad_story.pdf\n"
		" contents: minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's "
		"BOBB. minsing's Man United beat estdside_gunn's Chelsea.",
		"file_name: gunn_chelsea_sad_story.pdf\n"
		" contents: estdside_gunn always loses. I love this story.",
	],
	"token_ja": [
		"ファイル名: jeffrey_tigers_sad_story.pdf\n"
		" 内容: The Kia Tigers lost the Korean Series this year and failed to win the "
		"championship. jeffrey went to gwangju to the Korean Series, but they lost",
		"ファイル名: jeffrey_tigers_sad_story.pdf\n 内容: there. I love this story.",
		"ファイル名: gunn_chelsea_sad_story.pdf\n"
		" 内容: minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's BOBB. "
		"minsing's Man United beat estdside_gunn's Chelsea.",
		"ファイル名: gunn_chelsea_sad_story.pdf\n 内容: estdside_gunn always loses. I love this story.",
	],
	"character_ko": [
		f"파일 제목: jeffrey_tigers_sad_story.pdf\n 내용: {character_expect_texts[0]}",
		f"파일 제목: jeffrey_tigers_sad_story.pdf\n 내용: {character_expect_texts[1]}",
		f"파일 제목: jeffrey_tigers_sad_story.pdf\n 내용: {character_expect_texts[2]}",
		f"파일 제목: gunn_chelsea_sad_story.pdf\n 내용: {character_expect_texts[3]}",
		f"파일 제목: gunn_chelsea_sad_story.pdf\n 내용: {character_expect_texts[4]}",
		f"파일 제목: gunn_chelsea_sad_story.pdf\n 내용: {character_expect_texts[5]}",
		f"파일 제목: gunn_chelsea_sad_story.pdf\n 내용: {character_expect_texts[6]}",
	],
	"character_eng": [
		f"file_name: jeffrey_tigers_sad_story.pdf\n contents: {character_expect_texts[0]}",
		f"file_name: jeffrey_tigers_sad_story.pdf\n contents: {character_expect_texts[1]}",
		f"file_name: jeffrey_tigers_sad_story.pdf\n contents: {character_expect_texts[2]}",
		f"file_name: gunn_chelsea_sad_story.pdf\n contents: {character_expect_texts[3]}",
		f"file_name: gunn_chelsea_sad_story.pdf\n contents: {character_expect_texts[4]}",
		f"file_name: gunn_chelsea_sad_story.pdf\n contents: {character_expect_texts[5]}",
		f"file_name: gunn_chelsea_sad_story.pdf\n contents: {character_expect_texts[6]}",
	],
	"character_ja": [
		f"ファイル名: jeffrey_tigers_sad_story.pdf\n 内容: {character_expect_texts[0]}",
		f"ファイル名: jeffrey_tigers_sad_story.pdf\n 内容: {character_expect_texts[1]}",
		f"ファイル名: jeffrey_tigers_sad_story.pdf\n 内容: {character_expect_texts[2]}",
		f"ファイル名: gunn_chelsea_sad_story.pdf\n 内容: {character_expect_texts[3]}",
		f"ファイル名: gunn_chelsea_sad_story.pdf\n 内容: {character_expect_texts[4]}",
		f"ファイル名: gunn_chelsea_sad_story.pdf\n 内容: {character_expect_texts[5]}",
		f"ファイル名: gunn_chelsea_sad_story.pdf\n 内容: {character_expect_texts[6]}",
	],
	"overlap": [
		"The Kia Tigers lost the Korean Series this year and failed to win the championship. jeffrey went to gwangju to the Korean Series, but they lost",
		"to the Korean Series, but they lost there. I love this story.",
		"minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's BOBB. minsing's Man United beat estdside_gunn's Chelsea.",
		"United beat estdside_gunn's Chelsea. estdside_gunn always loses. I love this story.",
	],
}

expect_token_path = [
	base_metadata[0]["path"],
	base_metadata[0]["path"],
	base_metadata[1]["path"],
	base_metadata[1]["path"],
]

expect_character_path = [
	base_metadata[0]["path"],
	base_metadata[0]["path"],
	base_metadata[0]["path"],
	base_metadata[1]["path"],
	base_metadata[1]["path"],
	base_metadata[1]["path"],
	base_metadata[1]["path"],
]

expect_token_idx = [(0, 142), (144, 168), (0, 118), (120, 165)]
expect_overlap_idx = [(0, 142), (108, 168), (0, 118), (83, 165)]
expect_character_idx = [
	(0, 81),
	(84, 148),
	(151, 168),
	(0, 66),
	(69, 117),
	(120, 145),
	(148, 165),
]


def check_chunk_result(doc_id, contents, path, start_end_idx, metadata):
	params = [
		(doc_id, list, str),
		(contents, list, str),
		(path, list, str),
		(start_end_idx, list, tuple),
		(metadata, list, dict),
	]

	for param, outer_type, inner_type in params:
		assert isinstance(param, outer_type)
		assert isinstance(param[0], inner_type)


def check_chunk_result_node(result_df):
	check_chunk_result(
		result_df["doc_id"].tolist(),
		result_df["contents"].tolist(),
		result_df["path"].tolist(),
		result_df["start_end_idx"].tolist(),
		result_df["metadata"].tolist(),
	)


def test_make_metadata_list():
	parsed_df = pd.DataFrame(
		{
			"texts": base_texts,
			"page": [1, 1],
			"last_modified_datetime": ["24-09-03", "24-09-06"],
			"path": [
				"jax/jeffrey_tigers_sad_story.pdf",
				"siu/gunn_chelsea_sad_story.pdf",
			],
		}
	)
	meta_lst = make_metadata_list(parsed_df)
	assert meta_lst == [
		{
			"last_modified_datetime": "24-09-03",
			"page": 1,
			"path": "jax/jeffrey_tigers_sad_story.pdf",
		},
		{
			"last_modified_datetime": "24-09-06",
			"page": 1,
			"path": "siu/gunn_chelsea_sad_story.pdf",
		},
	]


def test_make_metadata_list_empty():
	parsed_df = pd.DataFrame({"texts": base_texts})
	meta_lst = make_metadata_list(parsed_df)
	assert meta_lst == [{}, {}]
