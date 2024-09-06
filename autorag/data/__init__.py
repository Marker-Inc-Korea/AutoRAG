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

parse_modules = {
	# PDF
	"pdfminer": PDFMinerLoader,
	"pdfplumber": PDFPlumberLoader,
	"pypdfium2": PyPDFium2Loader,
	"pypdf": PyPDFLoader,
	"pymupdf": PyMuPDFLoader,
	"unstructuredpdf": UnstructuredPDFLoader,
	"upstagelayoutanalysis": UpstageLayoutAnalysisLoader,
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
}
