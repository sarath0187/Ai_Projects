from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, TypedDict

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
from ddgs import DDGS

# ====================== API KEY ======================
os.environ["GROQ_API_KEY"] = "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"   # ← put your key here
# =====================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
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
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No web search results found."
        return "\n\n".join(
            [f"**{r.get('title', '')}**\n{r.get('body', '')}" for r in results]
        )
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def rag_tool(
    query: str,
    thread_id: Annotated[str, InjectedState("thread_id")],
) -> str:
    """Search the uploaded PDF for answers matching the user's query."""
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
    system_message = SystemMessage(
        content=(
            "You are a smart and helpful AI assistant.\n\n"
            "You have access to these tools:\n"
            "- search_tool: Use this for any current news, recent events, or real-time information.\n"
            "- rag_tool: Use this only when the user asks about an uploaded PDF.\n"
            "- calculator: Use for math questions.\n\n"
            "Very Important Rules:\n"
            "1. When the user asks for news or current information → ALWAYS call search_tool first.\n"
            "2. After you get the result from search_tool, you MUST write a clear, complete, and well-written final answer.\n"
            "3. Never reply with just one word, '?', '!', or incomplete sentences.\n"
            "4. Summarize the search results nicely and give useful information to the user.\n"
            "5. If the search tool returns results, use them to answer properly."
        )
    )

    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
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
