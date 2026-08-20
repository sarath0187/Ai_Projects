import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from back import chatbot, ingest_pdf, thread_has_document


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])
thread_key = str(st.session_state["thread_id"])


# Sidebar Layout
st.sidebar.title("Multi Utility AI Assistant")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# Thread Selector Dropdown
threads = st.session_state["chat_threads"][::-1]
selected_thread = st.sidebar.selectbox(
    "Select Conversation",
    options=threads,
    index=threads.index(st.session_state["thread_id"]),
)

# Switch thread if user selects a different one
if selected_thread != st.session_state["thread_id"]:
    st.session_state["thread_id"] = selected_thread
    thread_key = str(selected_thread)
    st.session_state["message_history"] = load_conversation(thread_key)
    st.rerun()

st.sidebar.markdown(f"**Current Thread ID:** `{thread_key}`")
st.sidebar.divider()

# PDF File Uploader
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])
if uploaded_file is not None:
    if st.sidebar.button("Index PDF", use_container_width=True):
        with st.spinner("Processing PDF..."):
            file_bytes = uploaded_file.read()
            msg = ingest_pdf(file_bytes, thread_key)
            st.sidebar.success(msg)

if thread_has_document(thread_key):
    st.sidebar.success("✅ PDF indexed for this thread.")
else:
    st.sidebar.info("ℹ️ No PDF uploaded for this thread yet.")


# Main Chat Interface
st.title("🤖 Multi-Utility AI Assistant")

# Sync local history with LangGraph state if empty
if "message_history" not in st.session_state or not st.session_state["message_history"]:
    st.session_state["message_history"] = load_conversation(thread_key)

# Render past messages (filtering out internal system/tool messages)
for msg in st.session_state["message_history"]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)

# Handle Chat Input
if prompt := st.chat_input("Ask a question, perform calculations, or query your PDF..."):
    # Render user prompt immediately
    with st.chat_message("user"):
        st.write(prompt)

    # Process prompt through LangGraph chatbot
    config = {"configurable": {"thread_id": thread_key}}
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.invoke(
                {"messages": [HumanMessage(content=prompt)], "thread_id": thread_key},
                config=config,
            )
            
            # Extract final AI response from state
            messages = response.get("messages", [])
            final_reply = ""
            for m in reversed(messages):
                if isinstance(m, AIMessage) and m.content:
                    final_reply = m.content
                    break
            
            st.write(final_reply)

    # Reload conversation history from checkpoint
    st.session_state["message_history"] = load_conversation(thread_key)
