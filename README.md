# 📚 InquireAI


A Python-based **Retrieval-Augmented Generation (RAG) chat application** built with **FastAPI**. Users can upload documents (PDF, TXT, DOCX, MD), and then ask questions — the app retrieves relevant chunks and provides AI-generated answers using **OpenAI** or **Ollama**.


---


## ✨ Features
- Upload multiple document types (`.pdf`, `.txt`, `.docx`, `.md`, `.csv`)
- Automatic text chunking & embeddings (`all-MiniLM-L6-v2` from `sentence-transformers`)
- Simple in-memory vector store (per session)
- RAG-powered chat over your documents
- Works with **OpenAI GPT models** or **local Ollama** models
- Minimal web UI (FastAPI + Tailwind + vanilla JS)


---


## ⚡ Quickstart


### 1. Clone the repo
```bash
git clone https://github.com/RahulSharma119/InquireAI.git
cd InquireAI
```


### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate # On Linux/Mac
venv\Scripts\activate # On Windows


pip install -r requirements.txt
```


### 3. Configure API keys
Edit `conf.ini` and add your keys/models, e.g.:


```ini
[LLM]
OPENAI_API_KEY = your_openai_api_key
OPENAI_MODEL = gpt-4o-mini

or

[LLM]
USE_OLLAMA = 0
OLLAMA_MODEL = llama3.1
```


- If `USE_OLLAMA=1`, the app will call a local Ollama model.
- Otherwise it will default to OpenAI using your API key.


### 4. Run the server
```bash
python -m uvicorn Main:app --reload --port 8000
```


### 5. Open in browser
👉 [http://localhost:8000/index.html](http://localhost:8000/index.html)

---

### 💡 Inspiration
This project is designed to make **your static documents conversational** — so you can query knowledge instantly with the power of **RAG + Generative AI**.