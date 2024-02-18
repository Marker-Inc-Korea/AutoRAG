import os
import shutil
from distutils.sysconfig import get_python_lib

import nltk


def pytest_sessionstart():
    nltk.download('punkt')  # Download for testing tree_summarize as an async process
    tests_dir = os.path.join(get_python_lib(), 'tests')
    if os.path.exists(tests_dir):
        shutil.rmtree(tests_dir)
