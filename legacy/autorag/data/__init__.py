import logging
from typing import Callable, List

from langchain_community.document_loaders import (
	BSHTMLLoader,
	CSVLoader,
	DirectoryLoader,
	JSONLoader,
	PDFMinerLoader,
	PDFPlumberLoader,
	PyMuPDFLoader,
	PyPDFium2Loader,
	PyPDFLoader,
	UnstructuredFileLoader,
	UnstructuredMarkdownLoader,
	UnstructuredPDFLoader,
	UnstructuredXMLLoader,
)
from langchain_text_splitters import (
	CharacterTextSplitter,
	KonlpyTextSplitter,
	RecursiveCharacterTextSplitter,
	SentenceTransformersTokenTextSplitter,
)
from llama_index.core.node_parser import (
	SemanticDoubleMergingSplitterNodeParser,
	SemanticSplitterNodeParser,
	SentenceSplitter,
	SentenceWindowNodeParser,
	SimpleFileNodeParser,
	TokenTextSplitter,
)

from autorag import LazyInit

logger = logging.getLogger("AutoRAG")


class UnstructuredLoader:
	def __init__(self, file_path_list: List[str], **kwargs):
		self._file_path_list = file_path_list
		self._kwargs = kwargs

	def load(self):
		documents = []
		for file_path in self._file_path_list:
			documents.extend(UnstructuredFileLoader(file_path, **self._kwargs).load())
		return documents


class UpstageLayoutAnalysisLoader:
	def __new__(cls, *args, **kwargs):
		loader_cls = None
		try:
			from langchain_upstage import (
				UpstageDocumentParseLoader as loader_cls,
			)
		except Exception:
			try:
				from langchain_upstage import UpstageLayoutAnalysisLoader as loader_cls
			except Exception as exc:
				raise ImportError(
					"The 'upstagedocumentparse' parser requires a compatible "
					"langchain-upstage installation. Install a version that supports "
					"your current langchain-core release."
				) from exc
		return loader_cls(*args, **kwargs)


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
