
# import streamlit as st
# from langchain_ollama import ChatOllama

# # Configure the website
# st.set_page_config(
#     page_title="AI Banking Assistant",
#     page_icon="🏦",
#     layout="centered"
# )

# st.title("🏦 AI Banking Assistant")
# st.caption("Powered by LangChain and Ollama")

# st.info(
#     "Ask me questions about loans, EMI, "
#     "interest rates, and general banking."
# )

# # Connect to the local LLM
# llm = ChatOllama(
#     model="llama3.2:3b",
#     temperature=0
# )

# # Store conversation history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display existing messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # Get a new question
# question = st.chat_input("Ask a banking question...")

# if question:
#     # Display and store the customer's question
#     st.session_state.messages.append({
#         "role": "user",
#         "content": question
#     })

#     with st.chat_message("user"):
#         st.write(question)

#     # Generate the AI response
#     with st.chat_message("assistant"):
#         with st.spinner("AI is thinking..."):
#             try:
#                 response = llm.invoke(question)
#                 answer = response.content

#             except Exception as e:
#                 answer = (
#                     "Unable to connect to the AI model. "
#                     "Please check that Ollama is running. "
#                     f"Details: {e}"
#                 )

#         st.write(answer)

#     # Save the response to the conversation
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": answer
#     })

# # Clear conversation history
# if st.button("Clear conversation"):
#     st.session_state.messages = []
#     st.rerun()


import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from rag import ask_banking_assistant

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

st.set_page_config(
    page_title="AI Banking Assistant",
    page_icon="🏦"
)

st.title("🏦 AI Banking Assistant")
st.caption("LangChain • Ollama • FAISS • RAG")

# PDF upload area
st.subheader("Banking documents")

uploaded_files = st.file_uploader(
    "Upload banking policy PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Create session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_retriever" not in st.session_state:
    st.session_state.active_retriever = None

if "upload_hash" not in st.session_state:
    st.session_state.upload_hash = None

# Process uploaded PDFs when the selection changes
current_hash = hashlib.sha256(
    b"".join(
        f.name.encode("utf-8") + f.getvalue()
        for f in uploaded_files
    )
).hexdigest() if uploaded_files else None

if current_hash != st.session_state.upload_hash:
    st.session_state.active_retriever = None
    st.session_state.upload_hash = current_hash

    if uploaded_files:
        try:
            with st.spinner("Processing uploaded PDFs..."):
                embeddings = OllamaEmbeddings(
                    model="nomic-embed-text"
                )

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                all_chunks = []

                # Load and split each uploaded PDF
                for uploaded_file in uploaded_files:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        pdf_path = Path(temp_dir) / uploaded_file.name
                        pdf_path.write_bytes(uploaded_file.getvalue())

                        loader = PyPDFLoader(str(pdf_path))
                        documents = loader.load()

                        for document in documents:
                            document.metadata["source"] = (
                                uploaded_file.name
                            )

                        all_chunks.extend(
                            splitter.split_documents(documents)
                        )

                if not all_chunks:
                    raise ValueError(
                        "No readable text was found in the PDFs."
                    )

                # Create an index for the uploaded PDFs
                uploaded_index = FAISS.from_documents(
                    all_chunks,
                    embeddings
                )

                # Combine with the existing sample FAQ index
                if (VECTORSTORE_PATH / "index.faiss").exists():
                    base_index = FAISS.load_local(
                        str(VECTORSTORE_PATH),
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    base_index.merge_from(uploaded_index)
                else:
                    base_index = uploaded_index

                st.session_state.active_retriever = (
                    base_index.as_retriever(
                        search_kwargs={"k": 3}
                    )
                )

            st.success(
                f"Processed {len(uploaded_files)} PDF file(s)."
            )

        except Exception as e:
            st.error(f"PDF processing failed: {e}")

# Show current documents
if uploaded_files:
    st.write("Uploaded files:")
    for file in uploaded_files:
        st.write(f"📄 {file.name}")
else:
    st.caption(
        "Using the existing banking FAQ index. "
        "Upload PDFs to add more searchable information."
    )

st.divider()

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.write(f"📄 {source}")

# Chat input
question = st.chat_input("Ask a banking question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, sources = ask_banking_assistant(
                    question,
                    st.session_state.active_retriever
                )
            except Exception as e:
                print("RAG error:", repr(e))
                answer = (
                    "Sorry, I couldn't process your question. "
                    "Please check the terminal for details."
                )
                sources = []

        st.write(answer)

        if sources:
            with st.expander("View sources"):
                for source in sources:
                    st.write(f"📄 {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
