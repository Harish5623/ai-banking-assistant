
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Find the project directory
BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_PATH = BASE_DIR / "documents" / "banking_faq.txt"
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

def main():
    # 1. Load the banking document
    print("Loading banking document...")

    loader = TextLoader(
        str(DOCUMENT_PATH),
        encoding="utf-8"
    )
    documents = loader.load()

    print("Documents loaded:", len(documents))

    # 2. Split the text into smaller chunks
    print("Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print("Chunks created:", len(chunks))

    # 3. Create embeddings using Ollama
    print("Creating embeddings with Ollama...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # 4. Store embeddings in FAISS
    print("Building FAISS vector database...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # 5. Save the vector database
    vectorstore.save_local(str(VECTORSTORE_PATH))

    print("FAISS index saved successfully!")
    print("Saved location:", VECTORSTORE_PATH)

if __name__ == "__main__":
    main()
