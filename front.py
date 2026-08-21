import uuid

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from back import (
    chatbot,
    ingest_pdf,
    thread_has_document,
)


# ============================================================
# THREAD FUNCTIONS
# ============================================================

def generate_thread_id():

    return str(uuid.uuid4())


def add_thread(thread_id):

    if "chat_threads" not in st.session_state:

        st.session_state["chat_threads"] = []

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"].append(
            thread_id
        )


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get(
        "messages",
        []
    )


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["message_history"] = []


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = (
        generate_thread_id()
    )


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = []


add_thread(
    st.session_state["thread_id"]
)


thread_key = str(
    st.session_state["thread_id"]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Multi Utility AI Assistant"
)


# New Chat
if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()


st.sidebar.divider()


# ============================================================
# THREAD SELECTOR
# ============================================================

threads = (
    st.session_state["chat_threads"][::-1]
)


selected_thread = st.sidebar.selectbox(
    "Select Conversation",
    options=threads,
    index=threads.index(
        st.session_state["thread_id"]
    ),
)


# User selected another conversation
if (
    selected_thread
    != st.session_state["thread_id"]
):

    st.session_state["thread_id"] = (
        selected_thread
    )

    thread_key = str(
        selected_thread
    )

    st.session_state["message_history"] = (
        load_conversation(thread_key)
    )

    st.rerun()


# Show current thread
st.sidebar.markdown(
    f"**Current Thread ID:** `{thread_key}`"
)


st.sidebar.divider()


# ============================================================
# PDF UPLOADER
# ============================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF Document",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.sidebar.button(
        "Index PDF",
        use_container_width=True
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            file_bytes = (
                uploaded_file.read()
            )

            msg = ingest_pdf(
                file_bytes,
                thread_key
            )

            st.sidebar.success(msg)


# PDF status
if thread_has_document(thread_key):

    st.sidebar.success(
        "✅ PDF indexed for this thread."
    )

else:

    st.sidebar.info(
        "ℹ️ No PDF uploaded for this thread yet."
    )


# ============================================================
# MAIN CHAT
# ============================================================

st.title(
    "🤖 Multi-Utility AI Assistant"
)


# Load history from LangGraph
if (
    "message_history" not in st.session_state
    or not st.session_state["message_history"]
):

    st.session_state["message_history"] = (
        load_conversation(thread_key)
    )


# ============================================================
# DISPLAY OLD MESSAGES
# ============================================================

for message in st.session_state[
    "message_history"
]:

    if isinstance(
        message,
        HumanMessage
    ):

        with st.chat_message("user"):

            st.write(
                message.content
            )

    elif (
        isinstance(message, AIMessage)
        and message.content
    ):

        with st.chat_message("assistant"):

            st.write(
                message.content
            )


# ============================================================
# CHAT INPUT
# ============================================================

if prompt := st.chat_input(
    "Ask a question, perform calculations, or query your PDF..."
):

    # Show user message immediately
    with st.chat_message("user"):

        st.write(prompt)


    # LangGraph configuration
    config = {
        "configurable": {
            "thread_id": thread_key
        }
    }


    # Run chatbot
    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            response = chatbot.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=prompt
                        )
                    ],
                    "thread_id": thread_key,
                },
                config=config,
            )


            # Find final AI response
            messages = response.get(
                "messages",
                []
            )

            final_reply = ""

            for message in reversed(messages):

                if (
                    isinstance(
                        message,
                        AIMessage
                    )
                    and message.content
                ):

                    final_reply = (
                        message.content
                    )

                    break


            st.write(final_reply)


    # Refresh history
    st.session_state[
        "message_history"
    ] = load_conversation(
        thread_key
    )
