import os
import shutil
from distutils.sysconfig import get_python_lib


def pytest_sessionstart():
    tests_dir = os.path.join(get_python_lib(), 'tests')
    if os.path.exists(tests_dir):
        shutil.rmtree(tests_dir)
