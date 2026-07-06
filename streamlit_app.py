"""
Sherlock Holmes RAG Chatbot — v2.1 (pilot-hardened).

Key improvements:
- Smart clarification: asks back, refuses, or picks confidently based on ambiguity
- Relevance floor: rejects low-score retrievals instead of guessing
- Sticky topic: follow-ups stay locked to the last-discussed story
- Query rewriting via CondensePlusContextChatEngine
- Story-tagged citations, suggested questions, error handling
"""

from typing import Any

import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import Memory
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

SYSTEM_PROMPT = (
    "You are a knowledgeable literary assistant specializing in "
    "'The Adventures of Sherlock Holmes' by Arthur Conan Doyle — a collection of TWELVE distinct short stories.\n\n"
    "Rules you MUST follow, in order:\n\n"
    "1. GROUNDING: Answer using ONLY the provided context passages. Never use outside knowledge, "
    "even if you 'know' the answer from training. If the passages do not contain the answer, say so plainly.\n\n"
    "2. AMBIGUITY: If the user's question is vague or could apply to multiple stories "
    "(e.g. 'tell me a story', 'what happens?', 'who is the villain?'), do ONE of the following:\n"
    "   (a) If the context clearly points to ONE story, name it confidently and summarize: "
    "'This appears to be from *The Red-Headed League*. In it...'\n"
    "   (b) If the context spans multiple stories, ask the user to specify which one, and list "
    "the story names from the retrieved context so they can choose.\n"
    "   (c) If nothing relevant was retrieved, say: 'I need a more specific question — try naming "
    "a character, story, or event from the Adventures.'\n\n"
    "3. FOLLOW-UPS: Vague follow-ups like 'tell me more', 'continue', 'what next', or 'go on' refer to "
    "the MOST RECENT story you were discussing. Do NOT switch stories on a follow-up. If the retrieved "
    "chunks are from a DIFFERENT story than the one you just discussed, ignore them and say: "
    "'The retrieved passages don't continue that thread — could you ask a more specific question about "
    "[previous story name]?'\n\n"
    "4. CITE THE STORY: When answering, always mention which of the twelve stories the answer comes from.\n\n"
    "5. LENGTH: Keep answers to 2-4 sentences unless the user explicitly asks for detail."
)

SUGGESTED_QUESTIONS = [
    "Who is Irene Adler and why does Holmes remember her?",
    "What was the trick behind The Red-Headed League?",
    "How did Holmes solve the Speckled Band mystery?",
    "Was Sherlock Holmes ever married?",
]

RELEVANCE_FLOOR = 0.35

st.set_page_config(page_title="Sherlock Holmes RAG", page_icon="🕵️", layout="centered")


@st.cache_resource(show_spinner="Loading Sherlock's case files...")
def load_resources(top_k: int, temperature: float) -> tuple[Any, Any]:
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
    import os
    persist_dir = "./vector_index_v2" if os.path.exists("./vector_index_v2") else "./vector_index"
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
    retriever = vector_index.as_retriever(similarity_top_k=top_k)
    return llm, retriever


with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 8, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    show_sources = st.checkbox("Show source citations", value=True)

    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.pop("memory", None)
        st.session_state.pop("history", None)
        st.session_state.pop("last_story", None)
        st.rerun()

    st.divider()
    st.caption(
        "Data: *The Adventures of Sherlock Holmes* by Arthur Conan Doyle "
        "(Project Gutenberg, public domain)."
    )


llm, retriever = load_resources(top_k, temperature)

if "memory" not in st.session_state:
    st.session_state.memory = Memory.from_defaults(token_limit=2000)
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "last_story" not in st.session_state:
    st.session_state.last_story = None


rag_bot = CondensePlusContextChatEngine.from_defaults(
    retriever=retriever,
    llm=llm,
    memory=st.session_state.memory,
    system_prompt=SYSTEM_PROMPT,
)


st.title("🕵️ Sherlock Holmes RAG")
st.caption("Ask me anything about *The Adventures of Sherlock Holmes*.")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📖 Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            story = src.get("story", "unknown")
            st.markdown(f"**Chunk {i}** — from *{story}* — relevance `{src['score']:.3f}`")
            st.text(src["text"])


if not st.session_state.history:
    st.markdown("**Try one of these to start:**")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(q, key=f"suggest_{i}", use_container_width=True):
            st.session_state.pending_prompt = q
            st.rerun()

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and show_sources:
            render_sources(turn.get("sources", []))


user_typed = st.chat_input("Elementary questions welcome...")
prompt = user_typed or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Consulting the case files..."):
                answer = rag_bot.chat(prompt)

            sources = []
            top_score = 0.0
            for node in getattr(answer, "source_nodes", []) or []:
                score = float(node.score) if node.score is not None else 0.0
                top_score = max(top_score, score)
                sources.append({
                    "score": score,
                    "story": node.metadata.get("story", "unknown"),
                    "text": node.text[:500] + ("..." if len(node.text) > 500 else ""),
                })

            if top_score < RELEVANCE_FLOOR and sources:
                response_text = (
                    "I couldn't find passages in the book that clearly match your question. "
                    "Try rephrasing — for example, name a specific character, story, or event "
                    "from *The Adventures of Sherlock Holmes*."
                )
            else:
                response_text = answer.response
                if sources:
                    st.session_state.last_story = sources[0]["story"]

            st.markdown(response_text)
            if show_sources:
                render_sources(sources)

            st.session_state.history.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources,
            })

        except Exception as e:
            error_msg = (
                "⚠️ Something went wrong reaching the language model. "
                "This is usually a temporary Groq API hiccup — try again in a moment."
            )
            st.error(error_msg)
            st.caption(f"Technical detail: `{type(e).__name__}: {str(e)[:200]}`")
            st.session_state.history.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
            })
