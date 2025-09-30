import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ------------------------------
# Helper Functions
# ------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_video_id(youtube_url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", youtube_url)
    return match.group(1) if match else None

def fetch_transcript(video_id: str) -> str:
    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id)
        raw_text = " ".join(chunk.text for chunk in fetched_transcript)
        return clean_text(raw_text)
    except TranscriptsDisabled:
        st.error("❌ Captions are disabled for this video.")
    except NoTranscriptFound:
        st.error("❌ Transcript not found for this video.")
    except Exception as e:
        st.error(f"⚠️ Error fetching transcript: {e}")
    return ""

def create_chunks(text: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(chunks, embeddings)

def build_qa_chain(retriever):
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', temperature=0.3)

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    return parallel_chain | prompt | model | parser


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="TalkTube 🗣️🎬", page_icon="🎥", layout="wide")

st.title("🎥 YouTube Video Q&A Assistant")
st.write("Enter a YouTube link, watch the video on the left, and chat with the assistant on the right.")

youtube_url = st.text_input("🔗 Enter YouTube Video Link:")

# Layout: 2 columns
col1, col2 = st.columns([1, 1])  # left = video, right = chat

with col1:
    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.video(f"https://www.youtube.com/watch?v={video_id}")

with col2:
    if youtube_url and st.button("Load Video Transcript"):
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("⚠️ Invalid YouTube URL.")
        else:
            with st.spinner("Fetching transcript and building vector store..."):
                transcript = fetch_transcript(video_id)
                if transcript:
                    chunks = create_chunks(transcript)
                    vector_store = create_vector_store(chunks)
                    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                    qa_chain = build_qa_chain(retriever)

                    st.session_state.qa_chain = qa_chain
                    st.session_state.messages = []  # reset chat
                    st.success("✅ Transcript loaded and cleaned! Start asking questions below.")


# ------------------------------
# Chat Interface (Full History)
# ------------------------------

    if "qa_chain" in st.session_state:
        st.subheader("💬 Chat with the Video")

        # Chat input pinned on top
        user_input = st.chat_input("Ask a question about the video...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Show ALL messages (newest on bottom)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ddd;
                        padding: 10px;
                        border-radius: 10px;
                        margin-bottom: 8px;
                    ">
                        👤 <b>You:</b> {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ddd;
                        padding: 10px;
                        border-radius: 10px;
                        margin-bottom: 8px;
                    ">
                        🗣️🎬 <b>TalkTube:</b> {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

