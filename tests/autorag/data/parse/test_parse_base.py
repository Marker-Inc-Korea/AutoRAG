import os
import pathlib
from datetime import datetime

from glob import glob

from autorag.data.parse.base import _add_last_modified_datetime
from autorag.data.utils.util import get_file_metadata

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resource_dir = os.path.join(root_dir, "resources")
result_dir = os.path.join(resource_dir, "test_results")
data_dir = os.path.join(resource_dir, "parse_data")

korean_text_glob = os.path.join(data_dir, "korean_text", "*")
korean_table_glob = os.path.join(data_dir, "korean_table", "only_table", "*")
eng_text_glob = os.path.join(data_dir, "eng_text", "*")
csv_glob = os.path.join(data_dir, "csv_data", "*")
json_glob = os.path.join(data_dir, "json_data", "*")
markdown_glob = os.path.join(data_dir, "markdown_data", "*")
html_glob = os.path.join(data_dir, "html_data", "*")
xml_glob = os.path.join(data_dir, "xml_data", "*")
all_files_glob = os.path.join(data_dir, "all_files", "*")
hybrid_glob = os.path.join(data_dir, "hybrid_data", "*")

single_pdf_path_list = glob(korean_text_glob)
multiple_pdf_data_list = glob(eng_text_glob)
csv_data_list = glob(csv_glob)
json_data_list = glob(json_glob)
markdown_data_list = glob(markdown_glob)
html_data_list = glob(html_glob)
xml_data_list = glob(xml_glob)
all_files_data_list = glob(all_files_glob)
hybrid_data_list = glob(hybrid_glob)
table_data_list = glob(korean_table_glob)

file_path_dict = {
	"single_pdf": single_pdf_path_list,
	"multiple_pdf": multiple_pdf_data_list,
	"csv": csv_data_list,
	"json": json_data_list,
	"markdown": markdown_data_list,
	"html": html_data_list,
	"xml": xml_data_list,
	"all_files_unstructured": [
		all_files_data_list[0],
		all_files_data_list[1],
		all_files_data_list[1],
		all_files_data_list[1],
		all_files_data_list[1],
		all_files_data_list[1],
		all_files_data_list[1],
		all_files_data_list[1],
		all_files_data_list[1],
		all_files_data_list[1],
		all_files_data_list[1],
	],
	"all_files_directory": all_files_data_list,
	"hybrid": hybrid_data_list,
	"hybrid_text": multiple_pdf_data_list,
	"hybrid_table": table_data_list,
}


def check_parse_result(texts, file_names, file_type):
	assert isinstance(texts, list)
	assert isinstance(texts[0], str)
	assert all([file_name in file_path_dict[file_type] for file_name in file_names])


def test_last_modified_datetime():
	result = (
		[
			"jeffrey love bali, kia tigers and Newjeans. But it's a top secret that he loves Newjeans. i love this "
			"story."
		],
		single_pdf_path_list,
		[-1],
	)
	result = _add_last_modified_datetime(result)

	date = get_file_metadata(single_pdf_path_list[0])["last_modified_datetime"]
	assert result[3] == [date]
