from typing import Dict


def add_gen_gt(row: Dict, new_gen_gt: str) -> Dict:
	if "generation_gt" in list(row.keys()):
		if isinstance(row["generation_gt"], list):
			row["generation_gt"].append(new_gen_gt)
		elif isinstance(row["generation_gt"], str):
			row["generation_gt"] = [row["generation_gt"], new_gen_gt]
		else:
			raise ValueError(
				"generation_gt should be either a string or a list of strings."
			)
		return row
	row["generation_gt"] = [new_gen_gt]
	return row
