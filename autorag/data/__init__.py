import logging
from typing import List, Callable

from langchain_community.document_loaders import (
	PDFMinerLoader,
	PDFPlumberLoader,
	PyPDFium2Loader,
	PyPDFLoader,
	PyMuPDFLoader,
	UnstructuredPDFLoader,
	CSVLoader,
	JSONLoader,
	UnstructuredMarkdownLoader,
	BSHTMLLoader,
	UnstructuredXMLLoader,
	DirectoryLoader,
)
from langchain_unstructured import UnstructuredLoader
from langchain_upstage import UpstageLayoutAnalysisLoader

from llama_index.core.node_parser import (
	TokenTextSplitter,
	SentenceSplitter,
	SentenceWindowNodeParser,
	SemanticSplitterNodeParser,
	SemanticDoubleMergingSplitterNodeParser,
	SimpleFileNodeParser,
)
from langchain.text_splitter import (
	RecursiveCharacterTextSplitter,
	CharacterTextSplitter,
	KonlpyTextSplitter,
	SentenceTransformersTokenTextSplitter,
)

from autorag import LazyInit

logger = logging.getLogger("AutoRAG")

parse_modules = {
	# PDF
	"pdfminer": PDFMinerLoader,
	"pdfplumber": PDFPlumberLoader,
	"pypdfium2": PyPDFium2Loader,
	"pypdf": PyPDFLoader,
	"pymupdf": PyMuPDFLoader,
	"unstructuredpdf": UnstructuredPDFLoader,
	# Common File Types
	# 1. CSV
	"csv": CSVLoader,
	# 2. JSON
	"json": JSONLoader,
	# 3. Markdown
	"unstructuredmarkdown": UnstructuredMarkdownLoader,
	# 4. HTML
	"bshtml": BSHTMLLoader,
	# 5. XML
	"unstructuredxml": UnstructuredXMLLoader,
	# 6. All files
	"directory": DirectoryLoader,
	"unstructured": UnstructuredLoader,
	"upstagedocumentparse": UpstageLayoutAnalysisLoader,
}

chunk_modules = {
	# Llama Index
	# Token
	"token": TokenTextSplitter,
	# Sentence
	"sentence": SentenceSplitter,
	# window
	"sentencewindow": SentenceWindowNodeParser,
	# Semantic
	"semantic_llama_index": SemanticSplitterNodeParser,
	"semanticdoublemerging": SemanticDoubleMergingSplitterNodeParser,
	# Simple
	"simplefile": SimpleFileNodeParser,
	# LangChain
	# Token
	"sentencetransformerstoken": SentenceTransformersTokenTextSplitter,
	# Character
	"recursivecharacter": RecursiveCharacterTextSplitter,
	"character": CharacterTextSplitter,
	# Sentence
	"konlpy": KonlpyTextSplitter,
}


def split_by_sentence_kiwi() -> Callable[[str], List[str]]:
	try:
		from kiwipiepy import Kiwi
	except ImportError:
		raise ImportError(
			"You need to install kiwipiepy to use 'ko_kiwi' tokenizer. "
			"Please install kiwipiepy by running 'pip install kiwipiepy'. "
			"Or install Korean version of AutoRAG by running 'pip install AutoRAG[ko]'."
		)
	kiwi = Kiwi()

	def split(text: str) -> List[str]:
		kiwi_result = kiwi.split_into_sents(text)
		sentences = list(map(lambda x: x.text, kiwi_result))

		return sentences

	return split


sentence_splitter_modules = {"kiwi": LazyInit(split_by_sentence_kiwi)}
