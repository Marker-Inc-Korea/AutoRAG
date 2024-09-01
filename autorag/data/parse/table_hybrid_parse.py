import itertools
import os
from typing import List, Tuple, Optional, Dict, Callable

from PyPDF2 import PdfFileReader, PdfFileWriter  # pip install 'PyPDF2<3.0'
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
	with_table_dir = os.path.join(pages_save_dir, "with_tables")
	os.makedirs(with_table_dir, exist_ok=True)
	without_table_dir = os.path.join(pages_save_dir, "without_tables")
	os.makedirs(without_table_dir, exist_ok=True)

	# Split PDF file into pages and Save PDFs with and without tables
	table_data_list, text_data_list = [], []
	for data_path in data_path_list:
		table_datas, text_datas = save_pdf_page_by_table(
			data_path, with_table_dir, without_table_dir
		)
		table_data_list.append(table_datas)
		text_data_list.append(text_datas)

	with_table_data_list = list(itertools.chain(*table_data_list))
	without_table_data_list = list(itertools.chain(*text_data_list))

	# Extract PDF without tables using text_parse_module
	table_parse_texts, table_parse_file_names = get_each_module_result(
		table_parse_module, table_params, with_table_data_list
	)

	# Extract PDF with tables using table_parse_module
	text_parse_texts, text_parse_file_names = get_each_module_result(
		text_parse_module, text_params, without_table_data_list
	)

	# Merge parsing results of PDFs with and without tables
	texts = table_parse_texts + text_parse_texts
	file_names = table_parse_file_names + text_parse_file_names

	# sort by file names
	file_names, texts = zip(*sorted(zip(file_names, texts)))

	return list(texts), list(file_names)


# Save PDFs with and without tables
def save_pdf_page_by_table(
	data_path: str, with_table_dir: str, without_table_dir: str
) -> Tuple[List[str], List[str]]:
	file_name = data_path.split("/")[-1].split(".pdf")[0]

	with_table_data_list, without_table_data_list = [], []

	# Read PDF file
	with open(data_path, "rb") as input_data:
		pdf_reader = PdfFileReader(input_data)
		num_pages = pdf_reader.getNumPages()

		for page_num in range(num_pages):
			pdf_writer = PdfFileWriter()
			pdf_writer.addPage(pdf_reader.getPage(page_num))

			# Table detection using pdfplumber
			with pdfplumber.open(data_path) as pdf:
				page = pdf.pages[page_num]
				tables = page.extract_tables()
				if tables:
					output_pdf_path = os.path.join(
						with_table_dir, f"{file_name}_page_{page_num + 1}.pdf"
					)
					with_table_data_list.append(output_pdf_path)
				else:
					output_pdf_path = os.path.join(
						without_table_dir, f"{file_name}_page_{page_num + 1}.pdf"
					)
					without_table_data_list.append(output_pdf_path)

				# Save page to PDF file
				with open(output_pdf_path, "wb") as output_path:
					pdf_writer.write(output_path)

	return with_table_data_list, without_table_data_list


def get_each_module_result(
	module: str, module_params: Dict, data_path_list: List[str]
) -> Tuple[List[str], List[str]]:
	module_params["module_type"] = module

	def get_param_combinations_pure(module_dict: Dict) -> Tuple[Callable, Dict]:
		module_instance = Module.from_dict(module_dict)
		return module_instance.module, module_instance.module_param

	text_parse_module_callable, text_parse_module_params = get_param_combinations_pure(
		module_params
	)
	original_text_module = text_parse_module_callable.__wrapped__
	texts, file_names = original_text_module(data_path_list, **text_parse_module_params)

	return texts, file_names
