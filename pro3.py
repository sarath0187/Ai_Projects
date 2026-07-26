import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("API")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_api_key,
)


def summarize(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    summaries = []

    for chunk in chunks:

        prompt = ChatPromptTemplate.from_template("""
You are an expert book summarizer.

Summarize the following part of the book.

{text}
""")

        message = prompt.invoke(
            {
                "text": chunk.page_content
            }
        )

        response = llm.invoke(message)

        summaries.append(response.content)

    combined_summary = "\n\n".join(summaries)

    final_prompt = ChatPromptTemplate.from_template("""
Below are summaries of different sections of a book.

Combine them into one complete, coherent summary.

{context}
""")

    final_message = final_prompt.invoke(
        {
            "context": combined_summary
        }
    )

    final_response = llm.invoke(final_message)

    st.markdown(final_response.content)


st.title("📚 Book Summarizer")

uploaded_file = st.file_uploader(
    "Upload a PDF Book",
    type=["pdf"]
)

if st.button("Generate Summary"):

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Generating Summary..."):
            summarize(uploaded_file)