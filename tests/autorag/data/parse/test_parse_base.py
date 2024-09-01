import os
import pathlib

from glob import glob

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resource_dir = os.path.join(root_dir, "resources")
result_dir = os.path.join(resource_dir, "test_results")
data_dir = os.path.join(resource_dir, "parse_data")

korean_text_glob = os.path.join(data_dir, "korean_text", "*")
korean_table_glob = os.path.join(data_dir, "korean_table", "*")
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

file_names_dict = {
	"single_pdf": ["korean_texts_two_page.pdf"],
	"multiple_pdf": ["baseball_1.pdf", "baseball_2.pdf"],
	"csv": ["csv_sample.csv"],
	"json": ["json_sample.json"],
	"markdown": ["markdown_sample.md"],
	"html": ["html_sample.html"],
	"xml": ["xml_sample.xml"],
	"all_files_unstructured": [
		"csv_sample.csv",
		"baseball_1.pdf",
		"baseball_1.pdf",
		"baseball_1.pdf",
		"baseball_1.pdf",
		"baseball_1.pdf",
		"baseball_1.pdf",
		"baseball_1.pdf",
		"baseball_1.pdf",
		"baseball_1.pdf",
		"baseball_1.pdf",
	],
	"all_files_directory": ["csv_sample.csv", "baseball_1.pdf"],
	"hybrid": ["nfl_rulebook_both_page_1.pdf", "nfl_rulebook_both_page_2.pdf"],
}


def check_parse_result(texts, file_names, file_type):
	assert isinstance(texts, list)
	assert isinstance(texts[0], str)
	if file_type == "json":
		assert texts == ["This is a sample JSON file"]
	assert all([file_name in file_names_dict[file_type] for file_name in file_names])
