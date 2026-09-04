

# Load API key safely from environment variable
# Run in terminal: export GROQ_API_KEY="your_new_key_here"

from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, TypedDict

from ddgs import DDGS

from langchain_community.document_loaders import PyPDFLoader
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
# GROQ API KEY
groq_api_key = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=groq_api_key,
)
# ============================================================

# Set your key in the terminal instead:
# export GROQ_API_KEY="your_new_key_here"



# ============================================================
# LLM
# ============================================================




# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ============================================================
# STORE PDF RETRIEVERS PER THREAD
# ============================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_pdf(file_bytes: bytes, thread_id: str) -> str:

    if not file_bytes:
        raise ValueError("No file content received.")

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split PDF into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(docs)

        # Create FAISS vector database
        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

        # Save retriever for this conversation
        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        return (
            f"Successfully indexed PDF "
            f"({len(chunks)} text chunks created)."
        )

    finally:

        # Delete temporary PDF
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# WEB SEARCH TOOL
# ============================================================

@tool
def search_tool(query: str) -> str:
    """
    Search the web for current events, recent information,
    news, weather, scores, and other time-sensitive information.
    """

    try:

        results = DDGS().text(
            query,
            max_results=3
        )

        if not results:
            return "No web search results found."

        formatted_results = []

        for i, result in enumerate(results, 1):

            title = result.get("title", "").strip()
            body = result.get("body", "").strip()
            href = result.get("href", "").strip()

            formatted_results.append(
                f"Result {i}:\n"
                f"Title: {title}\n"
                f"Content: {body}\n"
                f"Source: {href}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:

        return f"Search failed: {str(e)}"


# ============================================================
# PDF RAG TOOL
# ============================================================

@tool
def rag_tool(
    query: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Search the uploaded PDF for information relevant
    to the user's question.
    """

    thread_id = state.get("thread_id", "")

    retriever = _THREAD_RETRIEVERS.get(
        str(thread_id)
    )

    if not retriever:

        return (
            "No PDF has been uploaded for this chat yet. "
            "Please upload a PDF first."
        )

    try:

        docs = retriever.invoke(query)

        if not docs:
            return (
                "No matching information was found "
                "in the uploaded PDF."
            )

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    except Exception as e:

        return f"PDF search failed: {str(e)}"


# ============================================================
# CALCULATOR TOOL
# ============================================================

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> str:
    """
    Perform basic calculations:
    add, sub, mul, div.
    """

    if operation == "add":

        return str(
            first_num + second_num
        )

    elif operation == "sub":

        return str(
            first_num - second_num
        )

    elif operation == "mul":

        return str(
            first_num * second_num
        )

    elif operation == "div":

        if second_num == 0:

            return "Error: Cannot divide by zero."

        return str(
            first_num / second_num
        )

    return "Unsupported operation."


# ============================================================
# TOOLS
# ============================================================

tools = [
    rag_tool,
    calculator,
    search_tool,
]


# Give tools to LLM
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    thread_id: str


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(state: ChatState):

    system_message = SystemMessage(
        content=(
            "You are an AI assistant equipped with external tools.\n\n"

            "IMPORTANT TOOL RULES:\n"

            "1. CURRENT INFORMATION:\n"
            "Whenever the user asks about news, current events, "
            "weather, sports scores, recent facts, today's information, "
            "or anything time-sensitive, you MUST use "
            "`search_tool` before answering.\n\n"

            "2. MATHEMATICS:\n"
            "Whenever the user asks you to calculate something, "
            "use the `calculator` tool.\n\n"

            "3. UPLOADED PDF:\n"
            "Whenever the user's question is about information "
            "contained in their uploaded PDF, use `rag_tool`.\n\n"

            "4. DO NOT GUESS CURRENT INFORMATION:\n"
            "Never answer current or real-time questions from memory. "
            "Use `search_tool` first.\n\n"

            "5. NORMAL QUESTIONS:\n"
            "For normal conversational questions that don't require "
            "a tool, answer directly."
        )
    )

    # Remove old SystemMessages
    user_and_ai_messages = [
        message
        for message in state["messages"]
        if not isinstance(message, SystemMessage)
    ]

    # Add our system message
    full_messages = [
        system_message,
        *user_and_ai_messages
    ]

    # Ask LLM
    response = llm_with_tools.invoke(
        full_messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(ChatState)


# Add nodes
graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    ToolNode(tools)
)


# START → chat
graph.add_edge(
    START,
    "chat_node"
)


# chat → tools OR END
graph.add_conditional_edges(
    "chat_node",
    tools_condition
)


# tools → chat
graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# SQLITE CHECKPOINTER
# ============================================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)


chatbot = graph.compile(
    checkpointer=SqliteSaver(conn)
)


# ============================================================
# CHECK WHETHER THREAD HAS PDF
# ============================================================

def thread_has_document(
    thread_id: str
) -> bool:

    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )
