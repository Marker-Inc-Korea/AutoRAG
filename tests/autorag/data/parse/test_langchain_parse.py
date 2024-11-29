from autorag.data.parse import langchain_parse

from tests.autorag.data.parse.test_parse_base import (
	single_pdf_path_list,
	multiple_pdf_data_list,
	csv_data_list,
	json_data_list,
	markdown_data_list,
	html_data_list,
	xml_data_list,
	all_files_data_list,
	korean_text_glob,
	eng_text_glob,
	csv_glob,
	json_glob,
	markdown_glob,
	html_glob,
	xml_glob,
	all_files_glob,
	check_parse_result,
)


def test_langchain_parse_single_pdf():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(
		single_pdf_path_list, parse_method="pdfminer"
	)
	check_parse_result(texts, path, "single_pdf")
	assert pages == [-1]


def test_langchain_parse_single_pdf_node():
	result_df = langchain_parse(
		korean_text_glob, file_type="pdf", parse_method="pdfminer"
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"single_pdf",
	)
	assert result_df["page"].tolist() == [-1]


def test_langchain_parse_single_pdf_pages():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(
		single_pdf_path_list, parse_method="pymupdf"
	)
	check_parse_result(texts, path, "single_pdf")
	assert pages == [1, 2]


def test_langchain_parse_single_pdf_pages_node():
	result_df = langchain_parse(
		korean_text_glob, file_type="pdf", parse_method="pymupdf"
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"single_pdf",
	)
	assert result_df["page"].tolist() == [1, 2]


def test_langchain_parse_multiple_pdf():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(
		multiple_pdf_data_list, parse_method="pdfminer"
	)
	check_parse_result(texts, path, "multiple_pdf")
	assert pages == [-1, -1]


def test_langchain_parse_multiple_pdf_node():
	result_df = langchain_parse(eng_text_glob, file_type="pdf", parse_method="pdfminer")
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"multiple_pdf",
	)
	assert result_df["page"].tolist() == [-1, -1]


def test_langchain_parse_multiple_pdf_pages():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(
		multiple_pdf_data_list, parse_method="pymupdf"
	)
	check_parse_result(texts, path, "multiple_pdf")
	assert pages == [1, 1]


def test_langchain_parse_multiple_pdf_pages_node():
	result_df = langchain_parse(eng_text_glob, file_type="pdf", parse_method="pymupdf")
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"multiple_pdf",
	)
	assert result_df["page"].tolist() == [1, 1]


def test_langchain_csv():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(csv_data_list, parse_method="csv")
	check_parse_result(texts, path, "csv")


def test_langchain_csv_node():
	result_df = langchain_parse(csv_glob, file_type="csv", parse_method="csv")
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"csv",
	)


def test_langchain_json():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(
		json_data_list, parse_method="json", jq_schema=".content"
	)
	check_parse_result(texts, path, "json")
	assert texts == ["This is a sample JSON file"]


def test_langchain_json_node():
	result_df = langchain_parse(
		json_glob, file_type="json", parse_method="json", jq_schema=".content"
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"json",
	)


def test_langchain_markdown():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(
		markdown_data_list, parse_method="unstructuredmarkdown"
	)
	check_parse_result(texts, path, "markdown")


def test_langchain_markdown_node():
	result_df = langchain_parse(
		markdown_glob, file_type="md", parse_method="unstructuredmarkdown"
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"markdown",
	)


def test_langchain_html():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(html_data_list, parse_method="bshtml")
	check_parse_result(texts, path, "html")


def test_langchain_html_node():
	result_df = langchain_parse(html_glob, file_type="html", parse_method="bshtml")
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"html",
	)


def test_langchain_xml():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(
		xml_data_list, parse_method="unstructuredxml"
	)
	check_parse_result(texts, path, "xml")


def test_langchain_xml_node():
	result_df = langchain_parse(
		xml_glob, file_type="xml", parse_method="unstructuredxml"
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"xml",
	)


def test_langchain_all_files_unstructured():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, path, pages = langchain_parse_original(
		all_files_data_list, parse_method="unstructured"
	)
	check_parse_result(texts, path, "all_files_unstructured")


def test_langchain_all_files_unstructured_node():
	result_df = langchain_parse(
		all_files_glob, file_type="all_files", parse_method="unstructured"
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"all_files_unstructured",
	)


def test_langchain_all_files_directory():
	langchain_parse_original = langchain_parse.__wrapped__
	path_split_list = all_files_glob.split("/")
	glob_path = path_split_list.pop()
	folder_path = "/".join(path_split_list)
	texts, path, pages = langchain_parse_original(
		all_files_data_list, path=folder_path, glob=glob_path, parse_method="directory"
	)
	check_parse_result(texts, path, "all_files_directory")


def test_langchain_all_files_directory_node():
	result_df = langchain_parse(
		all_files_glob, file_type="all_files", parse_method="directory"
	)
	check_parse_result(
		result_df["texts"].tolist(), result_df["path"].tolist(), "all_files_directory"
	)
