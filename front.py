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


# Session State
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])
thread_key = str(st.session_state["thread_id"])
threads = st.session_state["chat_threads"][::-1]
selected_thread = None


# Sidebar
st.sidebar.title("Multi Utility AI Assistant")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

if thread_has_document(thread_key):
    st.sidebar.success("✅ A PDF is indexed and ready for this thread.")
else:
    st.sidebar.info("ℹ️ No PDF uploaded for this conversation yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    if st.sidebar.button("Index PDF", use_container_width=True):
        with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
            try:
                result_msg = ingest_pdf(
                    file_bytes=uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                )
                status_box.update(label="✅ PDF Indexed Successfully", state="complete", expanded=False)
                st.sidebar.success(result_msg)
                st.rerun()
            except Exception as e:
                status_box.update(label="❌ Ingestion Failed", state="error", expanded=False)
                st.sidebar.error(f"Error indexing PDF: {str(e)}")

st.sidebar.divider()
st.sidebar.subheader("Past Conversations")

if not threads:
    st.sidebar.write("No previous chats found.")
else:
    for t_id in threads:
        label = f"💬 {t_id[:8]}..." if t_id != thread_key else f"👉 {t_id[:8]}... (Active)"
        if st.sidebar.button(label, key=f"side-thread-{t_id}"):
            selected_thread = t_id


# Main Chat
st.title("AI Assistant & Document Reader")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask a question, upload a PDF, or run calculations...")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "recursion_limit": 20,
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}
        final_answer = ""
        tool_was_used = False

        for message_chunk, metadata in chatbot.stream(
            {
                "messages": [HumanMessage(content=user_input)],
                "thread_id": thread_key,
            },
            config=CONFIG,
            stream_mode="messages",
        ):
            if isinstance(message_chunk, ToolMessage):
                tool_was_used = True
                tool_name = getattr(message_chunk, "name", "Tool")
                if status_holder["box"] is None:
                    status_holder["box"] = st.status(f"🔧 Running `{tool_name}`...", expanded=True)
                else:
                    status_holder["box"].update(
                        label=f"🔧 Running `{tool_name}`...",
                        state="running",
                        expanded=True,
                    )

            if isinstance(message_chunk, AIMessage) and message_chunk.content:
                content = message_chunk.content.strip()
                # Ignore junk like "?", "!", ".", etc.
                if len(content) > 2:
                    final_answer = content

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

        if final_answer:
            st.markdown(final_answer)
        else:
            st.warning("No response generated. Please try again.")

    st.session_state["message_history"].append(
        {"role": "assistant", "content": final_answer}
    )


# Load past conversation
if selected_thread and selected_thread != thread_key:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            temp_messages.append({"role": "assistant", "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.rerun()
