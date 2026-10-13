
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

# Connect to the same embedding model used during ingestion
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Load our previously saved FAISS index
vectorstore = FAISS.load_local(
    str(VECTORSTORE_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)

# Create a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Test a customer question
question = "What documents are needed for a home loan?"

print("\nCustomer question:")
print(question)

print("\nRelevant document passages:")

results = retriever.invoke(question)

for i, document in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(document.page_content)
    print("Source:", document.metadata.get("source", "Unknown"))
