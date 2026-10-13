
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

response = llm.invoke(
    "Explain EMI in simple words."
)

print(response.content)
