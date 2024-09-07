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
}


def check_chunk_result(doc_id, metadata):
	assert isinstance(doc_id, list)
	assert isinstance(doc_id[0], str)
	assert isinstance(metadata, list)
	assert isinstance(metadata[0], dict)


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
