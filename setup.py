from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='RAGround',
    version='0.0.1',
    description='Automatically Evaluate RAG pipelines with "your own data". Find optimal baseline for new RAG product.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Marker-Inc',
    author_email='vkehfdl1@gmail.com',
    keywords=['RAG', 'RAGround', 'raground', 'rag-evaluation', 'evaluation', 'rag-ops', 'mlops'],
    python_requires='>=3.8',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    url="https://github.com/Marker-Inc-Korea/RAGround",
    license='Apache License 2.0',
    py_modules=[splitext(basename(path))[0] for path in glob('./*.py')],
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'pgvector',
        'tiktoken',
        'openai>=1.0.0',
        'llama-index>=0.9.28',
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ]
)
