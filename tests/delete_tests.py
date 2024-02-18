import os
import shutil
from distutils.sysconfig import get_python_lib


def main():
    tests_dir = os.path.join(get_python_lib(), 'tests')
    if os.path.exists(tests_dir):
        shutil.rmtree(tests_dir)


if __name__ == "__main__":
    main()
