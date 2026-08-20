from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, TypedDict

from duckduckgo_search import DDGS  # Fixed import name
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

# Load API key safely from environment variable
# Run in terminal: export GROQ_API_KEY="your_new_key_here"
os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,  # Slight temperature boost helps stop repetitive short outputs
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

_THREAD_RETRIEVERS: Dict[str, Any] = {}


def ingest_pdf(file_bytes: bytes, thread_id: str) -> str:
    if not file_bytes:
        raise ValueError("No file content received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        return f"Successfully indexed PDF ({len(chunks)} text chunks created)."
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@tool
def search_tool(query: str) -> str:
    """Search the web for current events and real-time information."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No web search results found."
        
        # Clean up formatting to prevent token disruption
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            formatted.append(f"Result {i}: {title} - {body}")
            
        return "\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def rag_tool(
    query: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Search the uploaded PDF for answers matching the user's query."""
    thread_id = state.get("thread_id", "")
    retriever = _THREAD_RETRIEVERS.get(str(thread_id))
    
    if not retriever:
        return "No PDF has been uploaded for this chat yet. Please upload one first."

    docs = retriever.invoke(query)
    if not docs:
        return "No matching information found in the uploaded PDF."

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


tools = [rag_tool, calculator, search_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str


def chat_node(state: ChatState):
    # System message explicitly instructing tool usage and dynamic temporal context
    system_message = SystemMessage(
        content=(
            "You are an AI assistant equipped with external tools.\n"
            "CRITICAL INSTRUCTION: You do not have real-time live web access on your own.\n"
            "1. Whenever the user asks for news, current events, weather, scores, or recent facts, "
            "you MUST invoke `search_tool(query=...)` first.\n"
            "2. If the user asks a math question, use `calculator`.\n"
            "3. If the user asks about an uploaded document, use `rag_tool`.\n"
            "Do NOT answer real-time or current queries from memory. Always call `search_tool`."
        )
    )

    # Clean existing system messages from history to prevent duplication
    user_and_ai_msgs = [
        msg for msg in state["messages"] if not isinstance(msg, SystemMessage)
    ]
    
    full_messages = [system_message] + user_and_ai_msgs
    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
chatbot = graph.compile(checkpointer=SqliteSaver(conn))


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS
