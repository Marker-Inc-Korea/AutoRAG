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
	texts, file_names = langchain_parse_original(
		single_pdf_path_list, parse_method="pdfminer"
	)
	check_parse_result(texts, file_names, "single_pdf")


def test_langchain_parse_single_pdf_node():
	result_df = langchain_parse(korean_text_glob, parse_method="pdfminer")
	check_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "single_pdf"
	)


def test_langchain_parse_multiple_pdf():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, file_names = langchain_parse_original(
		multiple_pdf_data_list, parse_method="pdfminer"
	)
	check_parse_result(texts, file_names, "multiple_pdf")


def test_langchain_parse_multiple_pdf_node():
	result_df = langchain_parse(eng_text_glob, parse_method="pdfminer")
	check_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "multiple_pdf"
	)


def test_langchain_csv():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, file_names = langchain_parse_original(csv_data_list, parse_method="csv")
	check_parse_result(texts, file_names, "csv")


def test_langchain_csv_node():
	result_df = langchain_parse(csv_glob, parse_method="csv")
	check_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "csv"
	)


def test_langchain_json():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, file_names = langchain_parse_original(
		json_data_list, parse_method="json", jq_schema=".content"
	)
	check_parse_result(texts, file_names, "json")


def test_langchain_json_node():
	result_df = langchain_parse(json_glob, parse_method="json", jq_schema=".content")
	check_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "json"
	)


def test_langchain_markdown():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, file_names = langchain_parse_original(
		markdown_data_list, parse_method="unstructuredmarkdown"
	)
	check_parse_result(texts, file_names, "markdown")


def test_langchain_markdown_node():
	result_df = langchain_parse(markdown_glob, parse_method="unstructuredmarkdown")
	check_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "markdown"
	)


def test_langchain_html():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, file_names = langchain_parse_original(html_data_list, parse_method="bshtml")
	check_parse_result(texts, file_names, "html")


def test_langchain_html_node():
	result_df = langchain_parse(html_glob, parse_method="bshtml")
	check_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "html"
	)


def test_langchain_xml():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, file_names = langchain_parse_original(
		xml_data_list, parse_method="unstructuredxml"
	)
	check_parse_result(texts, file_names, "xml")


def test_langchain_xml_node():
	result_df = langchain_parse(xml_glob, parse_method="unstructuredxml")
	check_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "xml"
	)


def test_langchain_all_files_unstructured():
	langchain_parse_original = langchain_parse.__wrapped__
	texts, file_names = langchain_parse_original(
		all_files_data_list, parse_method="unstructured"
	)
	check_parse_result(texts, file_names, "all_files_unstructured")


def test_langchain_all_files_unstructured_node():
	result_df = langchain_parse(all_files_glob, parse_method="unstructured")
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["file_name"].tolist(),
		"all_files_unstructured",
	)


def test_langchain_all_files_directory():
	langchain_parse_original = langchain_parse.__wrapped__
	path_split_list = all_files_glob.split("/")
	glob_path = path_split_list.pop()
	folder_path = "/".join(path_split_list)
	texts, file_names = langchain_parse_original(
		all_files_data_list, path=folder_path, glob=glob_path, parse_method="directory"
	)
	check_parse_result(texts, file_names, "all_files_directory")


def test_langchain_all_files_directory_node():
	result_df = langchain_parse(all_files_glob, parse_method="directory")
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["file_name"].tolist(),
		"all_files_directory",
	)
