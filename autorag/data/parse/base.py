import functools
import logging
from glob import glob
from typing import Tuple, List, Optional

from autorag.utils import result_to_dataframe

logger = logging.getLogger("AutoRAG")


def parser_node(func):
	@functools.wraps(func)
	@result_to_dataframe(["texts", "file_name"])
	def wrapper(
		data_path_glob: str, parse_method: Optional[str] = None, **kwargs
	) -> Tuple[List[str], List[str]]:
		logger.info(f"Running parser - {func.__name__} module...")

		data_path_list = glob(data_path_glob)

		if func.__name__ == "langchain_parse":
			parse_method = parse_method.lower()
			if parse_method == "directory":
				path_split_list = data_path_glob.split("/")
				glob_path = path_split_list.pop()
				folder_path = "/".join(path_split_list)
				kwargs.update({"glob": glob_path, "path": folder_path})
				result = func(
					data_path_list=data_path_list, parse_method=parse_method, **kwargs
				)
			else:
				result = func(
					data_path_list=data_path_list, parse_method=parse_method, **kwargs
				)
		elif func.__name__ in ["clova_ocr", "llama_parse"]:
			result = func(data_path_list=data_path_list, **kwargs)
		else:
			raise ValueError(f"Unsupported module_type: {func.__name__}")
		return result

	return wrapper
