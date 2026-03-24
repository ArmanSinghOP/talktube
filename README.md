# TalkTube — YouTube Video Q&A Assistant

🔗 Live App: https://talktube-20.streamlit.app/  
🔗 GitHub: https://github.com/ArmanSinghOP/talktube

---

## 🚧 Problem

Students spend hours watching long YouTube lectures but struggle to:
- Quickly extract key insights
- Revisit specific concepts
- Ask questions about video content

Existing solutions either:
- Require manual note-taking
- Provide generic summaries (not query-based)
- Lack context-grounded answers

**Goal:**  
Build a system that allows users to *interact with video content like a conversation*, ensuring answers are strictly grounded in the video itself.

---

## 💡 Approach

I built a **Retrieval-Augmented Generation (RAG)** system over YouTube transcripts.

### Pipeline:
1. Extract transcript using `youtube-transcript-api`
2. Clean and normalize text
3. Chunk text using `RecursiveCharacterTextSplitter`
4. Generate embeddings using Google Generative AI
5. Store vectors in **FAISS**
6. Retrieve top-k relevant chunks
7. Pass context to LLM for grounded response generation

### Key Principle:
> The model is strictly constrained to answer ONLY from retrieved transcript context

---

## 🔁 Iterations

### v1 — Basic Q&A
- Directly passed full transcript to LLM
- ❌ Slow and token inefficient
- ❌ Hallucinations

### v2 — Chunking + Embeddings
- Introduced chunking + FAISS
- ✅ Faster retrieval
- ❌ Still occasional irrelevant answers

### v3 — RAG with Prompt Constraints (Current)
- Added strict prompt:
  - “Answer ONLY from context”
  - “Say I don’t know if not found”
- Tuned:
  - chunk size = 1000
  - overlap = 200
  - top-k retrieval = 4
- ✅ High accuracy
- ✅ Minimal hallucination

---

## 🧠 Key Design Choices

### 1. RAG over Fine-tuning
- Keeps system lightweight
- Works on ANY video without retraining

### 2. FAISS for Vector Search
- Fast in-memory similarity search
- Scalable for future persistence

### 3. Google Generative AI
- Cost-efficient and fast inference
- Good performance for contextual QA

### 4. Chunking Strategy
- Balanced context vs performance
- Overlap ensures continuity

### 5. Streamlit UI
- Rapid prototyping
- Clean chat + video interface

---

## ⏱️ Time Commitment

- Duration: ~3–4 weeks
- Daily effort: 2–4 hours/day

Breakdown:
- Week 1: Core pipeline (transcripts + LLM)
- Week 2: RAG + FAISS integration
- Week 3: UI + prompt engineering + testing
- Week 4: Optimization + deployment

---

## ⚙️ Features

- Ask questions about any YouTube video
- Context-aware answers grounded in transcript
- Interactive chat interface with history
- Video + chat side-by-side UI
- Fast semantic retrieval using FAISS

---

## 🛠️ Tech Stack

- Python
- LangChain
- Google Generative AI
- FAISS
- Streamlit
- YouTube Transcript API

---

## 🚀 Future Improvements

- Multi-video knowledge base
- Persistent vector database
- Support for non-captioned videos (via ASR)
- Streaming responses
- Better ranking (re-ranking models)

---

## 📂 Project Setup

1. Clone the repo:

```bash
git clone https://github.com/ArmanSinghOP/talktube.git
cd talktube
```

2. Create and activate a Python virtual environment (recommended):

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

3. Install dependencies. Example `pip` command (you can also use `requirements.txt` if provided):

```bash
pip install streamlit youtube-transcript-api langchain langchain-google-genai langchain-community faiss-cpu python-dotenv
```

> Note: Installing FAISS on some platforms can be tricky. If `faiss-cpu` fails, check the FAISS project docs for platform-specific instructions.

4. Create a `.env` file in the project root and add your Google API key (used by `langchain_google_genai`):

```
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
```

(Alternatively store API keys using Streamlit secrets or environment variables managed by your deployment platform.)

5. Run the app:

```bash
streamlit run yt_chatbot.py
```

Open the Streamlit URL (usually `http://localhost:8501`) in your browser.

---

## How to Use

1. Paste a public YouTube video URL into the input box.
2. Click **Load Video Transcript**.
   - The app will fetch the transcript (if available), clean it, split it into chunks, compute embeddings, and store them in a FAISS index.
3. Ask questions in the chat input. The assistant will retrieve the top-k relevant chunks (k=4 by default) and answer using only those chunks.

UI layout: left column plays the video, right column contains the transcript loader and chat interface.

---

---

## 📬 Contact

For any queries, feel free to reach out via GitHub.
