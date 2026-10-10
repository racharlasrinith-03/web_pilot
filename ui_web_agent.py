
import os
import hashlib
import uuid
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="WebPilot AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLING ----------------

st.markdown("""
<style>
.stApp {
    background: #f7f8fc;
    color: #202536;
}
header[data-testid="stHeader"] {
    background: #f7f8fc;
}
.block-container {
    max-width: 1000px;
    padding-top: 1.5rem;
    padding-bottom: 5rem;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8eaf1;
}
.brand {
    font-size: 1.65rem;
    font-weight: 800;
    color: #4f46e5;
}
.tagline {
    color: #777f91;
    font-size: 0.9rem;
    margin-top: 4px;
}
.hero {
    text-align: center;
    padding: 30px 10px 20px;
}
.hero-icon {
    font-size: 2.5rem;
}
.hero h1 {
    font-size: 2.25rem;
    font-weight: 750;
    color: #202536;
    margin: 8px 0;
}
.hero p {
    font-size: 1rem;
    color: #777f91;
}
.url-card {
    background: #ffffff;
    border: 1px solid #e3e6ef;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(35, 45, 80, 0.04);
    margin: 10px 0 22px;
}
div.stButton > button,
div.stFormSubmitButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
}
div.stButton > button[kind="primary"],
div.stFormSubmitButton > button[kind="primary"] {
    background: #4f46e5;
    border-color: #4f46e5;
    color: white;
}
div[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e9ebf2;
    border-radius: 16px;
    padding: 15px;
    margin-bottom: 12px;
}
.feature {
    background: #ffffff;
    border: 1px solid #e8eaf1;
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    min-height: 100px;
}
.feature-icon {
    font-size: 1.35rem;
}
.feature-title {
    font-weight: 650;
    color: #30374b;
    margin-top: 6px;
    font-size: 0.9rem;
}
.feature-desc {
    color: #82899a;
    font-size: 0.78rem;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------

if "agent" not in st.session_state:
    st.session_state.agent = None

if "active_url" not in st.session_state:
    st.session_state.active_url = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ---------------- MODELS ----------------

@st.cache_resource(show_spinner=False)
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_tokens=400
    )

    return llm, embeddings


# ---------------- WEBSITE AGENT ----------------

def create_web_agent(url):
    # Hide internal function name and show a friendly message
    with st.spinner("Preparing AI assistant..."):
        llm, embeddings = load_models()

    collection_name = "web_" + hashlib.sha256(
        url.encode()
    ).hexdigest()[:32]

    db = Chroma(
        collection_name=collection_name,
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    if db._collection.count() == 0:
        with st.spinner("Reading website for the first time..."):
            docs = WebBaseLoader(url).load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=50
            )

            chunks = splitter.split_documents(docs)

            if not chunks:
                raise ValueError(
                    "No readable content found on this page."
                )

            db.add_documents(chunks)

    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    @tool
    def search_website(question: str) -> str:
        """Search the connected webpage for content, features,
        explanations, and summaries."""

        docs = retriever.invoke(question)

        if not docs:
            return "No relevant information found on the webpage."

        return "\n\n".join(
            doc.page_content for doc in docs
        )

    return create_agent(
        model=llm,
        tools=[search_website],
        checkpointer=MemorySaver(),
        system_prompt=f"""
You are WebPilot, a helpful website assistant.

Selected website: {url}

Developed By Racharla Srinth
Instructions:
- Never ask the user for the URL again.
- Search the website for questions about its content and features.
- Base website answers on retrieved information.
- Be honest when information is missing.
- Answer unrelated general questions directly.
- Remember earlier messages and understand follow-up questions.
- Give clear, concise answers.
"""
    )


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown(
        '<div class="brand">🌐 WebPilot</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="tagline">Your AI website companion</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("#### Connected website")

    if st.session_state.active_url:
        st.success("Website connected")
        st.caption(st.session_state.active_url)
    else:
        st.caption("No website connected yet.")

    st.markdown("---")
    st.markdown("#### What you can do")
    st.markdown(
        "- Summarize webpages\n"
        "- Discover key features\n"
        "- Ask technical questions\n"
        "- Ask follow-up questions"
    )

    st.markdown("---")
    st.caption("Powered by LangChain + LangGraph")


# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="hero-icon">✳️</div>
    <h1> Web Pilot, intelligently.</h1>
    <p>Connect a webpage and turn its content into a conversation.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- URL CONNECTOR ----------------

st.markdown('<div class="url-card">', unsafe_allow_html=True)
st.markdown("### 🔗 Connect a webpage")
st.caption("Paste a public webpage URL to get started.")

with st.form("connect_website"):
    url_input = st.text_input(
        "Website URL",
        value=st.session_state.active_url,
        placeholder="https://example.com",
        label_visibility="collapsed"
    )

    connect = st.form_submit_button(
        "Connect website →",
        type="primary",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

if connect:
    url = url_input.strip()

    if not url.startswith(("https://", "http://")):
        st.error("Please enter a valid URL starting with https:// or http://")

    elif not os.getenv("GROQ_API_KEY"):
        st.error("Add GROQ_API_KEY to your .env file.")

    else:
        try:
            if (
                st.session_state.agent is not None
                and st.session_state.active_url == url
            ):
                st.success("This website is already connected.")
            else:
                agent = create_web_agent(url)

                st.session_state.agent = agent
                st.session_state.active_url = url
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())

                st.success("Website connected successfully!")

            st.rerun()

        except Exception as e:
            st.error(f"Unable to connect: {e}")


# ---------------- FEATURES ----------------

if not st.session_state.messages and not st.session_state.agent:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature">
            <div class="feature-icon">🔎</div>
            <div class="feature-title">Smart search</div>
            <div class="feature-desc">Find useful page content</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Chat memory</div>
            <div class="feature-desc">Ask natural follow-ups</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Simple answers</div>
            <div class="feature-desc">Understand content faster</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### Try asking")
    st.markdown(
        "What is this page about? · Summarize the content · "
        "What are the main features?"
    )


# ---------------- CHAT ----------------

if st.session_state.agent:
    st.markdown("### 💬 Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about this webpage...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = st.session_state.agent.invoke(
                        {
                            "messages": [
                                {"role": "user", "content": question}
                            ]
                        },
                        config={
                            "configurable": {
                                "thread_id": st.session_state.thread_id
                            }
                        }
                    )

                    final_message = next(
                        (
                            msg for msg in reversed(result["messages"])
                            if msg.type == "ai" and not msg.tool_calls
                        ),
                        None
                    )

                    answer = (
                        final_message.content
                        if final_message
                        else "Sorry, I could not generate an answer."
                    )

                    if not isinstance(answer, str):
                        answer = str(answer)

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")

else:
    st.info("Connect a website above to activate your AI chat.")
