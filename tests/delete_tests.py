import os
import shutil
from distutils.sysconfig import get_python_lib


def main():
    tests_dir = os.path.join(get_python_lib(), 'tests')
    print(f"Github Actions: {os.getenv('GITHUB_ACTIONS')}")
    print(get_python_lib())
    if os.path.exists(tests_dir):
        shutil.rmtree(tests_dir)
    else:
        print("Directory does not exist")


if __name__ == "__main__":
    main()
