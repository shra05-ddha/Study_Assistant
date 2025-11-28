import os
import streamlit as st

from utils import extract_text_from_pdf
from agents import (
    explanation_agent,
    summarization_agent,
    quiz_agent,
    rag_answer_agent,
)
from retriever import (
    create_vector_store_from_text,
    retrieve_relevant_chunks,
)

from threading import Lock

# module-level lock persists across reruns in the same process and prevents
# multiple parallel request handlers from running at the same time.
if "_app_request_lock" not in globals():
    _app_request_lock = Lock()


# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Study Assistant (Multi-Agent)", layout="wide")
st.title("📚 StudySphere: Multi-Agent AI Assistant")

# -------------------------------------------------------
# SIDEBAR — PDF Upload + Feature Select
# -------------------------------------------------------
# ensure session keys exist before any widgets are created
st.session_state.setdefault("pdf_text", "")
st.session_state.setdefault("explain_history", [])
st.session_state.setdefault("rag_history", [])
st.session_state.setdefault("outputs", [])  # stores summary/quiz outputs
st.session_state.setdefault("llm_busy", False)
# Use uid-based widget keys so we can reset the input AFTER model returns
# without modifying the original widget key (avoids StreamlitAPIException).
st.session_state.setdefault("explain_input_uid", 0)
st.session_state.setdefault("rag_input_uid", 0)

st.sidebar.header("📄 Upload Notes")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])


if uploaded_pdf is not None:
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_pdf.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    st.sidebar.success("PDF uploaded successfully")

    # Extract and store persistently in session_state so other tabs can use it
    extracted = extract_text_from_pdf(pdf_path)
    st.session_state["pdf_text"] = extracted
    st.sidebar.info("PDF text extracted.")

    # Index for retrieval (creates/updates chroma local DB)
    create_vector_store_from_text(st.session_state["pdf_text"])
    st.sidebar.success("Notes indexed for RAG.")

st.sidebar.markdown("---")

feature = st.sidebar.radio(
    "Choose an action:",
    ["Explain Concept", "Summarize Notes", "Generate Quiz", "Chat with Notes (RAG)"],
)

# -------------------------------------------------------
# MAIN AREA — different UI per feature
# -------------------------------------------------------

st.markdown("---")

if feature == "Explain Concept":
    st.subheader("🧠 Explain Any Concept")

    # Chat history shown above the input (newest at bottom)
    for item in st.session_state["explain_history"]:
        role = item.get("role")
        text = item.get("text")
        if role == "user":
            st.markdown(f"You: {text}")
        else:
            st.markdown(f"Assistant: {text}")

    # form-based input at the bottom — do NOT clear before model replies.
    # We manually clear the widget inside the submit handler after the model returns.
    with st.form(key="explain_form", clear_on_submit=False):
        explain_key = f"explain_input_field_{st.session_state['explain_input_uid']}"
        question = st.text_area("Enter your question", key=explain_key, height=80)
        submitted = st.form_submit_button("Send")

        if submitted:
            if not question or not question.strip():
                st.warning("Please enter a question.")
            elif st.session_state.llm_busy:
                st.warning("A request is already running. Please wait for it to finish.")
            else:
                # Try to acquire the module-level request lock so we do not run
                # concurrent LLM requests in parallel (prevents out-of-order replies).
                # Block until the module-level lock is available so requests are
                # processed in-order and do not finish out of turn.
                _app_request_lock.acquire()
                processed = False
                try:
                    st.session_state.llm_busy = True
                    # capture input so it isn't affected by form clearing.
                    input_text = question.strip()
                    st.session_state["explain_history"].append({"role": "user", "text": input_text})
                    with st.spinner("Thinking with Llama..."):
                        reply = explanation_agent(input_text)
                    st.session_state["explain_history"].append({"role": "assistant", "text": reply})

                    # now clear the input widget by advancing the uid so the
                    # next render creates a fresh widget (safe — avoids modifying
                    # existing widget-backed session_key directly)
                    st.session_state["explain_input_uid"] += 1
                    processed = True
                finally:
                    st.session_state.llm_busy = False
                    _app_request_lock.release()
                if processed:
                    # Force an immediate rerun so the assistant reply is visible
                    st.rerun()

elif feature == "Summarize Notes":
    st.subheader("📝 Summarize Uploaded Notes")

    if not st.session_state["pdf_text"]:
        st.info("Upload a PDF file first from the sidebar.")
    else:
        if st.button("Summarize Notes"):
            # Acquire the module lock (blocking) so summarization runs in-order
            _app_request_lock.acquire()
            processed = False
            try:
                st.session_state.llm_busy = True
                with st.spinner("Summarizing using Llama..."):
                    summary = summarization_agent(st.session_state["pdf_text"])

                st.session_state["outputs"].append({"tag": "summary", "text": summary})
                st.success("Summary created — see below.")
            finally:
                st.session_state.llm_busy = False
                _app_request_lock.release()
            if processed:
                st.rerun()

    # show the latest summary result(s)
    for out in st.session_state["outputs"]:
        if out.get("tag") == "summary":
            st.markdown("Summary")
            st.write(out.get("text"))

elif feature == "Generate Quiz":
    st.subheader("🧩 Generate a Quiz")

    content_input = st.text_area(
        "Enter content manually OR leave empty to use uploaded PDF"
    )

    num_q = st.slider("Number of questions", 1, 10, 5)
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])

    if st.button("Generate Quiz"):
        # Block until the module lock is free so quiz generation runs after earlier requests
        _app_request_lock.acquire()
        processed = False
        try:
                content_to_use = content_input.strip() or st.session_state["pdf_text"]

                if not content_to_use:
                    st.warning("Upload a PDF or enter content.")
                else:
                    st.session_state.llm_busy = True
                    with st.spinner("Generating quiz..."):
                        quiz = quiz_agent(content_to_use, num=num_q, difficulty=difficulty)

                    st.session_state["outputs"].append({"tag": "quiz", "text": quiz})
        finally:
            st.session_state.llm_busy = False
            _app_request_lock.release()
        if processed:
            st.rerun()

    # Show generated quizzes
    for out in st.session_state["outputs"]:
        if out.get("tag") == "quiz":
            st.markdown("Generated Quiz")
            st.write(out.get("text"))

elif feature == "Chat with Notes (RAG)":
    st.subheader("💬 Ask Questions Based on PDF Notes")

    if not st.session_state["pdf_text"]:
        st.info("Upload a PDF first.")
    else:
        # show rag chat history
        for item in st.session_state["rag_history"]:
            if item.get("role") == "user":
                st.markdown(f"You: {item.get('text')}")
            else:
                st.markdown(f"Assistant: {item.get('text')}")

        # RAG input form (do not clear until model replies)
        with st.form(key="rag_form", clear_on_submit=False):
            rag_key = f"rag_input_field_{st.session_state['rag_input_uid']}"
            query = st.text_input("Enter your question about the notes", key=rag_key)
            submitted = st.form_submit_button("Ask")

            if submitted:
                if not query or not query.strip():
                    st.warning("Enter a question.")
                else:
                    # Block until we can run the request to guarantee ordered replies
                    _app_request_lock.acquire()
                    processed = False
                    try:
                        st.session_state.llm_busy = True
                        q = query.strip()
                        st.session_state["rag_history"].append({"role": "user", "text": q})
                        with st.spinner("Searching notes + Llama reasoning..."):
                            chunks = retrieve_relevant_chunks(q, k=3)
                            context = "\n\n".join([c.page_content for c in chunks])
                            answer = rag_answer_agent(q, context)

                        st.session_state["rag_history"].append({"role": "assistant", "text": answer})
                        # clear the widget AFTER reply by advancing uid
                        st.session_state["rag_input_uid"] += 1
                        processed = True
                    finally:
                        st.session_state.llm_busy = False
                        _app_request_lock.release()
                    if processed:
                        st.rerun()