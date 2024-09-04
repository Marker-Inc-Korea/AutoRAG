import os
import pathlib

import pandas as pd

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
base_file_names = ["jeffrey_tigers_sad_story.pdf", "gunn_chelsea_sad_story.pdf"]

parsed_result = pd.read_parquet(os.path.join(data_dir, "sample_parsed.parquet"))

expect_texts = {
	"original": [
		"The Kia Tigers lost the Korean Series this year and failed to win the championship. jeffrey went to "
		"gwangju to the Korean",
		"Series, but they lost there. I love this story.",
		"minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's BOBB. minsing's Man United beat",
		"estdside_gunn's Chelsea. estdside_gunn always loses. I love this story.",
	],
	"korean": [
		"파일 제목: jeffrey_tigers_sad_story.pdf\n 내용: The Kia Tigers lost the Korean Series this year and failed to win the championship.",
		"파일 제목: jeffrey_tigers_sad_story.pdf\n 내용: jeffrey went to gwangju to the Korean Series, but they lost",
		"파일 제목: jeffrey_tigers_sad_story.pdf\n 내용: there. I love this story.",
		"파일 제목: gunn_chelsea_sad_story.pdf\n 내용: minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's",
		"파일 제목: gunn_chelsea_sad_story.pdf\n 내용: BOBB. minsing's Man United beat estdside_gunn's Chelsea.",
		"파일 제목: gunn_chelsea_sad_story.pdf\n 내용: estdside_gunn always loses. I love this story.",
	],
	"english": [
		"file_name: jeffrey_tigers_sad_story.pdf\n contents: The Kia Tigers lost the Korean Series this year and failed to win the championship.",
		"file_name: jeffrey_tigers_sad_story.pdf\n contents: jeffrey went to gwangju to the Korean Series, but they lost",
		"file_name: jeffrey_tigers_sad_story.pdf\n contents: there. I love this story.",
		"file_name: gunn_chelsea_sad_story.pdf\n contents: minsing's Real Madrid were crushed by Ulsan Hyundai of Korea's",
		"file_name: gunn_chelsea_sad_story.pdf\n contents: BOBB. minsing's Man United beat estdside_gunn's Chelsea.",
		"file_name: gunn_chelsea_sad_story.pdf\n contents: estdside_gunn always loses. I love this story.",
	],
}


def check_chunk_result(doc_id, metadata):
	assert isinstance(doc_id, list)
	assert isinstance(doc_id[0], str)
	assert isinstance(metadata, list)
	assert isinstance(metadata[0], dict)
