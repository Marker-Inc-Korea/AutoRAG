import os
import tempfile
from glob import glob
from typing import List, Tuple, Dict

from PyPDF2 import PdfFileReader, PdfFileWriter
import pdfplumber

from autorag.support import get_support_modules
from autorag.data.parse.base import parser_node


@parser_node
def table_hybrid_parse(
	data_path_list: List[str],
	text_parse_module: str,
	text_params: Dict,
	table_parse_module: str,
	table_params: Dict,
) -> Tuple[List[str], List[str], List[int]]:
	"""
	Parse documents to use table_hybrid_parse method.
	The table_hybrid_parse method is a hybrid method that combines the parsing results of PDFs with and without tables.
	It splits the PDF file into pages, separates pages with and without tables, and then parses and merges the results.

	:param data_path_list: The list of data paths to parse.
	:param text_parse_module: The text parsing module to use. The type should be a string.
	:param text_params: The extra parameters for the text parsing module. The type should be a dictionary.
	:param table_parse_module: The table parsing module to use. The type should be a string.
	:param table_params: The extra parameters for the table parsing module. The type should be a dictionary.
	:return: tuple of lists containing the parsed texts, path and pages.
	"""
	# make save folder directory
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as save_dir:
		text_dir = os.path.join(save_dir, "text")
		table_dir = os.path.join(save_dir, "table")

		os.makedirs(text_dir, exist_ok=True)
		os.makedirs(table_dir, exist_ok=True)

		# Split PDF file into pages and Save PDFs with and without tables
		path_map_dict_lst = [
			save_page_by_table(data_path, text_dir, table_dir)
			for data_path in data_path_list
		]
		path_map_dict = {k: v for d in path_map_dict_lst for k, v in d.items()}

		# Extract text pages
		table_results, table_file_path = get_each_module_result(
			table_parse_module, table_params, os.path.join(table_dir, "*")
		)

		# Extract table pages
		text_results, text_file_path = get_each_module_result(
			text_parse_module, text_params, os.path.join(text_dir, "*")
		)

		# Merge parsing results of PDFs with and without tables
		texts = table_results + text_results
		temp_path_lst = table_file_path + text_file_path

		# Sort by file names
		temp_path_lst, texts = zip(*sorted(zip(temp_path_lst, texts)))

		# get original file path
		path = list(map(lambda temp_path: path_map_dict[temp_path], temp_path_lst))

		# get pages
		pages = list(map(lambda x: get_page_from_path(x), temp_path_lst))

		return list(texts), path, pages


# Save PDFs with and without tables
def save_page_by_table(data_path: str, text_dir: str, table_dir: str) -> Dict[str, str]:
	file_name = os.path.basename(data_path).split(".pdf")[0]

	with open(data_path, "rb") as input_data:
		pdf_reader = PdfFileReader(input_data)
		num_pages = pdf_reader.getNumPages()

		path_map_dict = {}
		for page_num in range(num_pages):
			output_pdf_path = _get_output_path(
				data_path, page_num, file_name, text_dir, table_dir
			)
			_save_single_page(pdf_reader, page_num, output_pdf_path)
			path_map_dict.update({output_pdf_path: data_path})

	return path_map_dict


def _get_output_path(
	data_path: str, page_num: int, file_name: str, text_dir: str, table_dir: str
) -> str:
	with pdfplumber.open(data_path) as pdf:
		page = pdf.pages[page_num]
		tables = page.extract_tables()
		directory = table_dir if tables else text_dir
		return os.path.join(directory, f"{file_name}_page_{page_num + 1}.pdf")


def _save_single_page(pdf_reader: PdfFileReader, page_num: int, output_pdf_path: str):
	pdf_writer = PdfFileWriter()
	pdf_writer.addPage(pdf_reader.getPage(page_num))

	with open(output_pdf_path, "wb") as output_file:
		pdf_writer.write(output_file)


def get_each_module_result(
	module: str, module_params: Dict, data_path_glob: str
) -> Tuple[List[str], List[str]]:
	module_params["module_type"] = module

	data_path_list = glob(data_path_glob)
	if not data_path_list:
		return [], []

	module_name = module_params.pop("module_type")
	module_callable = get_support_modules(module_name)
	module_original = module_callable.__wrapped__
	texts, path, _ = module_original(data_path_list, **module_params)

	return texts, path


def get_page_from_path(file_path: str) -> int:
	file_name = os.path.basename(file_path)
	split_result = file_name.rsplit("_page_", -1)
	page_number_with_extension = split_result[1]
	page_number, _ = page_number_with_extension.split(".")

	return int(page_number)
