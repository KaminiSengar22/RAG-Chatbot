# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2d2f3e; }
    .chat-title { font-size: 2rem; font-weight: 700; color: #ffffff; margin-bottom: 0.2rem; }
    .chat-subtitle { color: #8b8fa8; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .source-box {
        background: #1e2130;
        border-left: 4px solid #6c63ff;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 13px;
        color: #c5c8d6;
    }
    .stat-card {
        background: #1e2130;
        border: 1px solid #2d2f3e;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
        color: #c5c8d6;
    }
    .stChatInput textarea { background-color: #1e2130 !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

VECTORSTORE_PATH = "vectorstore/"

@st.cache_resource(show_spinner="Loading AI model...")
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate.from_template("""You are a helpful AI assistant. Use the following context to answer the question accurately.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 RAG Chatbot")
    st.markdown("Ask questions about your loaded documents.")
    st.divider()

    st.markdown("**⚙️ Configuration**")
    st.markdown('<div class="stat-card">🧠 <b>LLM:</b> LLaMA 3.3 70B<br>🔎 <b>Embeddings:</b> MiniLM-L6-v2<br>📦 <b>Vector DB:</b> FAISS<br>🌡️ <b>Temperature:</b> 0.2</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**📊 Session Stats**")
    total_msgs = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    st.markdown(f'<div class="stat-card">💬 Questions asked: <b>{total_msgs}</b></div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**💡 Sample Questions**")
    sample_questions = [
        "What is RAG?",
        "How does FAISS work?",
        "What are embeddings?",
        "Explain LangChain",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill"] = q
            st.rerun()

# ── Main Area ─────────────────────────────────────────────
st.markdown('<div class="chat-title">🤖 RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">Powered by LLaMA 3.3 · FAISS · LangChain · Groq</div>', unsafe_allow_html=True)
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 View sources used"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(
                        f'<div class="source-box"><b>Source {i}:</b><br>{doc.page_content[:250]}...</div>',
                        unsafe_allow_html=True
                    )

# Handle sample question prefill
if "prefill" in st.session_state:
    prefill_q = st.session_state.pop("prefill")
    st.session_state.messages.append({"role": "user", "content": prefill_q})
    with st.chat_message("user"):
        st.markdown(prefill_q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain, retriever = load_chain()
            answer = chain.invoke(prefill_q)
            sources = retriever.invoke(prefill_q)
        st.markdown(answer)
        with st.expander("📄 View sources used"):
            for i, doc in enumerate(sources, 1):
                st.markdown(
                    f'<div class="source-box"><b>Source {i}:</b><br>{doc.page_content[:250]}...</div>',
                    unsafe_allow_html=True
                )
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# Handle typed input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain, retriever = load_chain()
            answer = chain.invoke(prompt)
            sources = retriever.invoke(prompt)
        st.markdown(answer)
        with st.expander("📄 View sources used"):
            for i, doc in enumerate(sources, 1):
                st.markdown(
                    f'<div class="source-box"><b>Source {i}:</b><br>{doc.page_content[:250]}...</div>',
                    unsafe_allow_html=True
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })