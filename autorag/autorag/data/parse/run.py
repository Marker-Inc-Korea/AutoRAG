import os
from typing import List, Callable, Dict
import pandas as pd
from glob import glob

from autorag.strategy import measure_speed
from autorag.data.utils.util import get_param_combinations

default_map = {
	"pdf": {
		"file_type": "pdf",
		"module_type": "langchain_parse",
		"parse_method": "pdfminer",
	},
	"csv": {
		"file_type": "csv",
		"module_type": "langchain_parse",
		"parse_method": "csv",
	},
	"md": {
		"file_type": "md",
		"module_type": "langchain_parse",
		"parse_method": "unstructuredmarkdown",
	},
	"html": {
		"file_type": "html",
		"module_type": "langchain_parse",
		"parse_method": "bshtml",
	},
	"xml": {
		"file_type": "xml",
		"module_type": "langchain_parse",
		"parse_method": "unstructuredxml",
	},
}


def run_parser(
	modules: List[Callable],
	module_params: List[Dict],
	data_path_glob: str,
	project_dir: str,
	all_files: bool,
):
	if not all_files:
		# Set the parsing module to default if it is a file type in paths but not set in YAML.
		data_path_list = glob(data_path_glob)
		if not data_path_list:
			raise FileNotFoundError(f"data does not exits in {data_path_glob}")

		file_types = set(
			[os.path.basename(data_path).split(".")[-1] for data_path in data_path_list]
		)
		set_file_types = set([module["file_type"] for module in module_params])

		# create a list of only those file_types that are in file_types but not in set_file_types
		missing_file_types = list(file_types - set_file_types)
		if list(set_file_types - file_types):
			raise ValueError(
				f"File types {list(set_file_types - file_types)} are not in the data path."
			)

		if missing_file_types:
			add_modules_list = []
			for missing_file_type in missing_file_types:
				if missing_file_type == "json":
					raise ValueError(
						"JSON file type must have a jq_schema so you must set it in the YAML file."
					)

				add_modules_list.append(default_map[missing_file_type])

			add_modules, add_params = get_param_combinations(add_modules_list)
			modules.extend(add_modules)
			module_params.extend(add_params)

	results, execution_times = zip(
		*map(
			lambda x: measure_speed(x[0], data_path_glob=data_path_glob, **x[1]),
			zip(modules, module_params),
		)
	)
	average_times = list(map(lambda x: x / len(results[0]), execution_times))

	# save results to parquet files
	if all_files:
		filepaths = list(
			map(
				lambda x: os.path.join(project_dir, f"{x}.parquet"), range(len(modules))
			)
		)
	else:
		filepaths = list(
			map(
				lambda x: os.path.join(project_dir, f"{x['file_type']}.parquet"),
				module_params,
			)
		)
	list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))
	filenames = list(map(lambda x: os.path.basename(x), filepaths))

	summary_df = pd.DataFrame(
		{
			"filename": filenames,
			"module_name": list(map(lambda module: module.__name__, modules)),
			"module_params": module_params,
			"execution_time": average_times,
		}
	)
	summary_df.to_csv(os.path.join(project_dir, "summary.csv"), index=False)

	# concat all parquet files here if not all_files.
	if not all_files:
		dataframes = [pd.read_parquet(file) for file in filepaths]
		combined_df = pd.concat(dataframes, ignore_index=True)
		combined_df.to_parquet(
			os.path.join(project_dir, "parsed_result.parquet"), index=False
		)

	return summary_df
