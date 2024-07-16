import utils
import ollama
import os

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

print("Loading documents...")
loader = DirectoryLoader(os.getenv("DOCUMENTS_DIR"), glob="*.pdf")
print("Loaded pdf files from", os.getenv("DOCUMENTS_DIR"))

documents = loader.load()
print("Number of documents:", len(documents))

# # print document context
# for doc in documents:
#     print(f"Document: {doc.metadata['source']}")
#     print(doc.page_content)
#     print("\n")

embeddings_model = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=embeddings_model, show_progress=True)
print("Embeddings model:", embeddings_model)

# text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="interquartile")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    add_start_index=True,
)

texts = text_splitter.split_documents(documents)
print("Number of texts:", len(texts))

vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding=embeddings,
    persist_directory=os.getenv("VECTORDATABASE_DIR"))

