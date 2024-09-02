import os
from glob import glob
from typing import List, Tuple, Optional, Dict, Callable

from PyPDF2 import PdfFileReader, PdfFileWriter
import pdfplumber

from autorag.data.parse.base import parser_node
from autorag.schema import Module


@parser_node
def table_hybrid_parse(
	data_path_list: List[str],
	text_parse_module: str,
	text_params: Dict,
	table_parse_module: str,
	table_params: Dict,
	pages_save_dir: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
	# make save folder directory
	pages_save_dir if pages_save_dir is not None else os.getcwd()
	os.makedirs(pages_save_dir, exist_ok=True)
	text_dir = os.path.join(pages_save_dir, "text")
	os.makedirs(text_dir, exist_ok=True)
	table_dir = os.path.join(pages_save_dir, "table")
	os.makedirs(table_dir, exist_ok=True)

	# Split PDF file into pages and Save PDFs with and without tables
	[save_page_by_table(data_path, text_dir, table_dir) for data_path in data_path_list]

	# Extract text pages
	table_results, table_file_names = get_each_module_result(
		table_parse_module, table_params, os.path.join(table_dir, "*")
	)

	# Extract table pages
	text_results, text_file_names = get_each_module_result(
		text_parse_module, text_params, os.path.join(text_dir, "*")
	)

	# Merge parsing results of PDFs with and without tables
	texts = table_results + text_results
	file_names = table_file_names + text_file_names

	# sort by file names
	file_names, texts = zip(*sorted(zip(file_names, texts)))

	return list(texts), list(file_names)


# Save PDFs with and without tables
def save_page_by_table(data_path: str, text_dir: str, table_dir: str):
	file_name = os.path.basename(data_path).split(".pdf")[0]

	with open(data_path, "rb") as input_data:
		pdf_reader = PdfFileReader(input_data)
		num_pages = pdf_reader.getNumPages()

		for page_num in range(num_pages):
			output_pdf_path = _get_output_path(
				data_path, page_num, file_name, text_dir, table_dir
			)
			_save_single_page(pdf_reader, page_num, output_pdf_path)


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
		raise FileNotFoundError(f"data does not exits in {data_path_glob}")

	def get_param_combinations_pure(module_dict: Dict) -> Tuple[Callable, Dict]:
		module_instance = Module.from_dict(module_dict)
		return module_instance.module, module_instance.module_param

	module_callable, module_params = get_param_combinations_pure(module_params)
	module_original = module_callable.__wrapped__
	texts, file_names = module_original(data_path_list, **module_params)

	return texts, file_names
