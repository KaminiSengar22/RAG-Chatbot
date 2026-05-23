# 🤖 RAG Chatbot — Retrieval Augmented Generation


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-chatbot-8pzzsskqpnnmxtvpfscndp.streamlit.app/)


A production-style AI chatbot that answers questions from your own documents using **Retrieval Augmented Generation (RAG)**. Built with LangChain, FAISS, Groq LLaMA 3.3, and Streamlit.

---

## 🚀 Features

- 📄 Query any PDF or text document using natural language
- 🔎 Semantic search using FAISS vector store
- 🧠 Powered by LLaMA 3.3 70B via Groq (free tier)
- 💬 Clean chat UI with history and source document display
- ⚡ Fully local embeddings using HuggingFace sentence-transformers
- 🗑️ Clear chat history and session stats in sidebar

---

## 🏗️ Architecture

User Question
↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
↓
FAISS Vector Store → Top-3 Relevant Chunks
↓
LLaMA 3.3 70B via Groq API
↓
Answer + Sources

---

## 🛠️ Tech Stack

| Component     | Tool                          |
|---------------|-------------------------------|
| Framework     | LangChain                     |
| Vector Store  | FAISS (local)                 |
| Embeddings    | HuggingFace all-MiniLM-L6-v2  |
| LLM           | Groq — LLaMA 3.3 70B          |
| UI            | Streamlit                     |
| Language      | Python 3.10+                  |

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
# Create a .env file and add:
# GROQ_API_KEY=your_key_here

# 5. Ingest documents
python ingest.py

# 6. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

rag-chatbot/
├── data/               ← Place your PDF or .txt files here
├── vectorstore/        ← FAISS index (auto-generated, gitignored)
├── ingest.py           ← Document loading & indexing pipeline
├── rag_chain.py        ← Core RAG chain logic
├── app.py              ← Streamlit web UI
├── requirements.txt    ← Python dependencies
└── .env                ← API keys (gitignored, never committed)

---

## 👩‍💻 Author

**Kamini Sengar** — AI/ML Engineer  
[GitHub](https://github.com/KaminiSengar22) • [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)