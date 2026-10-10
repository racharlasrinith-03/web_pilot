
import os
import hashlib
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

# Check API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Add GROQ_API_KEY to your .env file.")

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=400
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Get website URL
url = input("Enter website URL: ").strip()

if not url.startswith(("https://", "http://")):
    raise ValueError("Enter a valid URL starting with http:// or https://")

# Create a separate collection for each website
collection_name = "web_" + hashlib.md5(
    url.encode()
).hexdigest()

# Initialize ChromaDB
db = Chroma(
    collection_name=collection_name,
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Index website only once
if db._collection.count() == 0:
    print("\nLoading website...")

    docs = WebBaseLoader(url).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No content found on this webpage.")

    db.add_documents(chunks)
    print("Website indexed successfully!")

else:
    print("Using existing website database.")

# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# Website search tool
@tool
def search_website(question: str) -> str:
    """Search the selected website for its content, features,
    explanations, summaries, and technical information."""

    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information was found on the website."

    return "\n\n".join(
        doc.page_content for doc in docs
    )


# LangGraph conversation memory
memory = MemorySaver()

# Create agent
agent = create_agent(
    model=llm,
    tools=[search_website],
    checkpointer=memory,
    system_prompt=f"""
You are a helpful Web Agent Chatbot.

The user has selected this website:
{url}
Developded by :Racharla Srinith
Instructions:
1. Never ask the user for the URL again.
2. Use search_website for questions about the selected webpage.
3. Search the website for its features, content, and summaries.
4. Use conversation history to understand follow-up questions.
5. Answer general questions directly when appropriate.
6. If website information is missing, say so honestly.
7. Give clear, simple, accurate answers.
"""
)

# Conversation configuration
config = {
    "configurable": {
        "thread_id": "web_agent_chat"
    }
}

print("\nWeb Agent Chatbot is ready!")
print("Ask questions about the website or general topics.")
print("Type 'exit' to stop.\n")

# Chat loop
while True:
    try:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Web Agent: Goodbye!")
            break

        if not question:
            continue

        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            config=config
        )

       
        answer = result["messages"][-1].content

        print(f"\nWeb Agent: {answer}\n")

    except KeyboardInterrupt:
        print("\n Web Agent: Goodbye!")
        break

    except Exception as e:
        print(f"\nError: {e}\n")
