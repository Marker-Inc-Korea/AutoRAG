import os
import pathlib

import pandas as pd

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resource_dir = os.path.join(root_dir, "resources")
data_dir = os.path.join(resource_dir, "chunk_data")

base_texts = ["", ""]

parsed_result = pd.read_parquet(os.path.join(data_dir, "sample_parsed.parquet"))
