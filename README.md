# TalkTube — YouTube Video Q&A Assistant

A lightweight Streamlit app that turns YouTube videos into an interactive Q&A experience. Paste a YouTube link, load the transcript, and ask natural-language questions — the assistant answers using only the transcript content.

Live App - https://talktube-20.streamlit.app/

---

## Features

- Play YouTube video in the left column while chatting on the right.
- Fetches captions/transcript using `youtube-transcript-api`.
- Cleans and chunks long transcripts with `RecursiveCharacterTextSplitter`.
- Embeds chunks with Google Generative AI embeddings and indexes with FAISS.
- Uses `langchain_google_genai`'s `ChatGoogleGenerativeAI` to answer user questions, constrained to transcript context.
- Simple, session-state backed chat UI (keeps full conversation history during the session).

---

## Quick Start

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

## Implementation Details & Flow

This section describes the important functions and the processing flow implemented in `app.py` (or whichever file you use):

1. **Async loop fix**

   - At the top of the app an `asyncio` event-loop guard runs to ensure Streamlit’s environment can create or reuse a running loop without crashing.

2. **extract_video_id(youtube_url)**

   - Uses a regex to extract the 11-character YouTube video id from standard links (`v=`) or `youtu.be` short links.

3. **fetch_transcript(video_id)**

   - Uses `YouTubeTranscriptApi().fetch(video_id)` to get captions.
   - Joins caption segments into a single text blob and calls `clean_text`.
   - Handles `TranscriptsDisabled` and `NoTranscriptFound` exceptions and surfaces friendly Streamlit error messages.

4. **clean_text(text)**

   - Replaces newlines with spaces, fixes common camelCase runs by inserting a period and space between lower->Upper transitions, collapses whitespace.

5. **Chunking**

   - `create_chunks(text, chunk_size=1000, chunk_overlap=200)` uses `RecursiveCharacterTextSplitter` to produce document chunks from the cleaned transcript. Adjust `chunk_size` and `chunk_overlap` if answers lack context or you want finer-grained retrieval.

6. **Embedding & Indexing**

   - `create_vector_store(chunks)` creates `GoogleGenerativeAIEmbeddings(model="models/embedding-001")` and builds a FAISS index with `FAISS.from_documents(chunks, embeddings)`.
   - The app currently builds the index in-memory each time you press **Load Video Transcript**; for production or faster loads you may persist FAISS indices to disk.

7. **QA Chain**

   - `build_qa_chain(retriever)` constructs a LangChain runnable pipeline:
     - `PromptTemplate` instructs the model to "Answer ONLY from the provided transcript context. If the context is insufficient, just say you don't know."
     - `ChatGoogleGenerativeAI` with `model='gemini-1.5-pro'` and `temperature=0.3` is used for final answer generation.
     - `RunnableParallel` is used so the retriever's results are formatted (`format_docs`) and passed as the `context` input to the prompt.
     - `StrOutputParser` returns the final text answer.

8. **Chat Interaction**

   - When the user asks a question, the app calls `qa_chain.invoke(user_input)` and appends the assistant reply to `st.session_state.messages`.
   - The app renders all messages in the session using simple HTML/CSS styling blocks.

---

## Configuration & Tips

- **Chunk size / overlap**: If answers are missing detail or the assistant incorrectly says "I don't know", increase `chunk_size` or the number of retrieved documents `k` (`search_kwargs={"k": 4}`) when creating the retriever.
- **Persist FAISS**: For repeated queries against the same video, save/load FAISS indexes to disk to avoid recomputing embeddings on every load.
- **Alternative embeddings/models**: If you prefer to use OpenAI embeddings or other providers, swap `GoogleGenerativeAIEmbeddings` and `ChatGoogleGenerativeAI` with the provider of your choice and update the prompt/model accordingly.
- **Streamlit secrets**: For deployment, use Streamlit Cloud secrets or the hosting platform’s env vars to store API keys securely — do NOT commit `.env` to source control.

---

## Troubleshooting & Common Issues

- **Transcript not found / disabled**: Many videos lack auto-captions or have captions disabled; `YouTubeTranscriptApi` will raise `NoTranscriptFound` or `TranscriptsDisabled`. Use another video or add subtitles to the video.
- **Invalid YouTube URL**: The extractor expects standard YouTube URLs or `youtu.be` short links. If you get `None` for `video_id`, verify the URL.
- **FAISS install failure**: On Windows or some Linux distributions, `faiss-cpu` may need extra build steps. Search FAISS installation instructions for your platform.
- **High latency / cost**: Generating embeddings and model calls incur API calls — expect some latency and potential API costs depending on your Google account and model usage.

---

## File Structure (suggested)

```
TalkTube/
├─ yt_chatbot.py         # main Streamlit app (your provided code)
├─ README.txt            # this file
├─ requirements.txt      # pip dependencies
├─ .env                  # local environment variables (NOT committed)
├─ .gitignore
└─ assets/               # optional: images, icons
```

---

## Example `requirements.txt` (suggested)

```
streamlit
youtube-transcript-api
langchain
langchain-google-genai
langchain-community
faiss-cpu
python-dotenv
```

Pin versions if you need reproducibility.

---

## Next Steps / Improvements

- Add support for videos in other languages.
- Persist and cache FAISS indexes to speed repeated loads.
- Add support for uploading a local transcript file (SRT/TXT) when YouTube captions are unavailable.
- Add streaming responses or typing indicators for better UX.
- Add authentication and rate-limiting for multi-user deployments.
- Add unit tests for core helper functions (ID extraction, cleaning, chunking).

---

## License & Credits

MIT License — feel free to fork and extend.  
Credits: built with `streamlit`, `youtube-transcript-api`, `langchain` and Google Generative AI.

---

## Contact

If you want help integrating features (FAISS persistence, deployment, or replacing Google embeddings), open an issue or contact me via the GitHub repo.

---
