import importlib
import importlib.util
import os
import shutil
import sys


def is_ubuntu() -> bool:
	try:
		with open("/etc/os-release") as f:
			for line in f:
				if line.startswith("NAME") and "Ubuntu" in line:
					return True
	except FileNotFoundError:
		pass
	return False


def is_github_action() -> bool:
	if not is_ubuntu():
		return False
	if importlib.util.find_spec("torch") is None:
		return True
	torch_cuda = importlib.import_module("torch.cuda")
	return not torch_cuda.is_available()


def main():
	paths = sys.path
	for path in paths:
		tests_dir = os.path.join(path, "tests")
		if os.path.exists(tests_dir):
			print(f"Deleting {tests_dir}")
			shutil.rmtree(tests_dir)


if __name__ == "__main__":
	main()
