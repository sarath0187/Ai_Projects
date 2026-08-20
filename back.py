from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition

# ============================================================
# Environment & Core Setup
# ============================================================

load_dotenv()

# Set environment variables directly in code
os.environ["GROQ_API_KEY"] = "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"

# Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Global dictionary mapping thread_id -> FAISS retriever
_THREAD_RETRIEVERS: Dict[str, Any] = {}


# ============================================================
# PDF Ingestion
# ============================================================

def ingest_pdf(file_bytes: bytes, thread_id: str) -> str:
    """
    Reads incoming PDF bytes, splits text into chunks,
    and indexes them into FAISS stored by thread_id.
    """
    if not file_bytes:
        raise ValueError("No file content received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Load and split PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        # Store in FAISS
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # Link retriever to thread
        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        return f"Successfully indexed PDF ({len(chunks)} text chunks created)."

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# Tools
# ============================================================

# DuckDuckGo Search Tool
from duckduckgo_search import DDGS
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search the web for current events and real-time information."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No web search results found."
        return "\n\n".join([f"**{r.get('title', '')}**\n{r.get('body', '')}" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def rag_tool(
    query: str,
    thread_id: Annotated[str, InjectedState("thread_id")],
) -> str:
    """
    Search the uploaded PDF for answers matching the user's query.
    The thread_id is automatically injected by LangGraph.
    """
    retriever = _THREAD_RETRIEVERS.get(str(thread_id))

    if not retriever:
        return "No PDF has been uploaded for this chat yet. Please upload one first."

    docs = retriever.invoke(query)

    if not docs:
        return "No matching information found in the uploaded PDF."

    # Return plain matching text directly
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """Perform basic calculations: add, sub, mul, div."""
    if operation == "add":
        return str(first_num + second_num)
    elif operation == "sub":
        return str(first_num - second_num)
    elif operation == "mul":
        return str(first_num * second_num)
    elif operation == "div":
        if second_num == 0:
            return "Error: Cannot divide by zero."
        return str(first_num / second_num)
    return "Unsupported operation."


# Bind tools (RAG + Calculator + DuckDuckGo Search) to the model
tools = [rag_tool, calculator, search_tool]
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# State & Execution Nodes
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str


def chat_node(state: ChatState):
   system_message = SystemMessage(
    content=(
        "You are a helpful AI assistant.\n"
        "If the user asks about an uploaded document, ALWAYS call rag_tool first.\n"
        "Use search_tool for real-time web info or current news.\n"
        "Use the calculator tool for arithmetic."
    )
)
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# ============================================================
# Graph & Memory Setup
# ============================================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# Persistent memory using SQLite
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
chatbot = graph.compile(checkpointer=SqliteSaver(conn))


# ============================================================
# Helper Functions
# ============================================================

def thread_has_document(thread_id: str) -> bool:
    """Check if a given thread already has an indexed PDF."""
    return str(thread_id) in _THREAD_RETRIEVERS
