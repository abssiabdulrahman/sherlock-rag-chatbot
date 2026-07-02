"""
Sherlock Holmes RAG Chatbot — Streamlit app.

Prerequisites:
1. Run `sherlock_rag_chatbot.ipynb` at least once to build `./vector_index/`.
2. Create `.streamlit/secrets.toml` with:
       GROQ_API_KEY = "your_key_here"
3. Run: `streamlit run streamlit_sherlock_app.py`
"""

from typing import Any

import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import Memory
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# ---------- Page config ----------

st.set_page_config(
    page_title="Sherlock Holmes RAG",
    page_icon="🕵️",
    layout="centered",
)


# ---------- Shared resources (cached across users) ----------

@st.cache_resource
def load_resources(top_k: int, temperature: float) -> tuple[Any, Any, list[ChatMessage]]:
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=512,
    )

    embeddings = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./embedding_model/",
    )

    storage_context = StorageContext.from_defaults(persist_dir="./vector_index")
    vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

    retriever = vector_index.as_retriever(similarity_top_k=top_k)

    prefix_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a knowledgeable literary assistant specializing in "
                "'The Adventures of Sherlock Holmes' by Arthur Conan Doyle."
            ),
        ),
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Answer questions using ONLY the provided context from the book. "
                "If the answer is not in the context, say so honestly rather than guessing."
            ),
        ),
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="Keep answers concise (2-4 sentences) unless asked for more detail.",
        ),
    ]

    return llm, retriever, prefix_messages


# ---------- Sidebar ----------

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Chunks retrieved per question (top-k)",
        min_value=1,
        max_value=8,
        value=3,
        help="More chunks = more context, but slower and noisier.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Low = factual and deterministic. High = creative.",
    )

    show_sources = st.checkbox("Show source citations", value=True)

    st.divider()

    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.pop("memory", None)
        st.session_state.pop("history", None)
        st.rerun()

    st.divider()
    st.caption(
        "Data: *The Adventures of Sherlock Holmes* by Arthur Conan Doyle "
        "(Project Gutenberg, public domain)."
    )


# ---------- Load resources with current settings ----------

llm, retriever, prefix_messages = load_resources(top_k, temperature)


# ---------- Per-session state ----------

if "memory" not in st.session_state:
    st.session_state.memory = Memory.from_defaults(token_limit=2000)

if "history" not in st.session_state:
    # We track our own display history so we can attach sources to each turn
    st.session_state.history = []  # list of {"role": str, "content": str, "sources": list[dict]}


# ---------- Chat engine ----------

rag_bot = ContextChatEngine(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    prefix_messages=prefix_messages,
)


# ---------- UI ----------

st.title("🕵️ Sherlock Holmes RAG")
st.caption("Ask me anything about *The Adventures of Sherlock Holmes*.")


# Replay previous turns
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander(f"📖 Sources ({len(turn['sources'])})"):
                for i, src in enumerate(turn["sources"], 1):
                    st.markdown(f"**Chunk {i}** — relevance: `{src['score']:.3f}`")
                    st.text(src["text"])


# New user message
if prompt := st.chat_input("Elementary questions welcome..."):

    st.session_state.history.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the case files..."):
            answer = rag_bot.chat(prompt)

        st.markdown(answer.response)

        # Collect source chunks used
        sources = []
        for node in getattr(answer, "source_nodes", []) or []:
            sources.append({
                "score": float(node.score) if node.score is not None else 0.0,
                "text": node.text[:500] + ("..." if len(node.text) > 500 else ""),
            })

        if show_sources and sources:
            with st.expander(f"📖 Sources ({len(sources)})"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}** — relevance: `{src['score']:.3f}`")
                    st.text(src["text"])

    st.session_state.history.append({
        "role": "assistant",
        "content": answer.response,
        "sources": sources,
    })
