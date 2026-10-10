# WebPilot AI

WebPilot AI is a website question-answering chatbot built with LangChain, LangGraph, Groq, Hugging Face embeddings, ChromaDB, and Streamlit. Users can connect a public webpage, ask questions about its content, request summaries, and ask follow-up questions in a conversational interface.

## Features

- **Website URL input:** Connect a publicly accessible webpage.
- **Website question answering:** Retrieve relevant page content to answer questions.
- **RAG pipeline:** Split page content into chunks, embed the chunks, and store them in ChromaDB.
- **Persistent vector storage:** Reuse indexed content for the same URL across app runs.
- **Agent tool use:** The agent can call a website-search tool when it needs page information.
- **Conversation memory:** LangGraph's in-memory checkpointer maintains context during the current application process.
- **General chat:** Answer questions unrelated to the selected webpage.
- **Streamlit UI:** A simple chat interface with website connection controls.

## Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- Groq API (`ChatGroq`)
- Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- `python-dotenv`

## Project Structure

```text
project/
├── app.py
├── .env
├── chroma_db/
└── requirements.txt
```

- `app.py` — Streamlit application and agent logic.
- `.env` — Stores the Groq API key. Do not commit this file.
- `chroma_db/` — Local vector database created at runtime.
- `requirements.txt` — Python dependencies.

## Prerequisites

- Python 3.10 or newer (use a version supported by your installed dependencies).
- A Groq API key.
- Internet access to fetch the selected webpage and call the Groq API.

## Installation

### 1. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -U streamlit langchain langchain-groq langchain-huggingface langchain-community langchain-chroma langchain-text-splitters langgraph sentence-transformers python-dotenv beautifulsoup4
```

### 3. Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Replace the placeholder with your own key. Never publish or commit your API key.

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, commonly `http://localhost:8501`.

## How to Use

1. Open WebPilot AI in your browser.
2. Paste a public webpage URL into the **Connect a webpage** field.
3. Select **Connect website** and wait for the page to be indexed if it is new.
4. Ask questions such as:
   - “What is this page about?”
   - “What are the main features?”
   - “Summarize this webpage.”
   - “Explain the second feature in more detail.”
5. Ask general questions unrelated to the webpage if needed.
6. Use **New conversation** to start a new conversation thread.

## How It Works

1. **Load webpage:** `WebBaseLoader` fetches the specified page.
2. **Split content:** `RecursiveCharacterTextSplitter` divides the page into chunks.
3. **Create embeddings:** Hugging Face's `all-MiniLM-L6-v2` converts chunks into vectors.
4. **Store vectors:** ChromaDB stores the vectors in `./chroma_db`.
5. **Retrieve context:** The `search_website` tool retrieves relevant chunks for a question.
6. **Generate an answer:** The Groq-powered agent uses retrieved content or responds to general questions.
7. **Maintain conversation state:** LangGraph's `MemorySaver` tracks conversation messages for the active thread.

## Notes and Limitations

- The application indexes the specific URL entered, not every page on the website.
- Some websites block automated requests or render their content with JavaScript, which a basic loader may not capture.
- A URL's ChromaDB collection is reused on later runs. If the webpage changes, the existing index will not automatically refresh.
- `MemorySaver` stores conversation state in memory. Conversation checkpoints are lost when the application process stops.
- General answers come from the language model and may not always be correct.
- The app should only fetch trusted, public URLs. For a public deployment, add URL validation and protections against requests to localhost, private IP addresses, and internal services.

## Security

- Keep `.env` out of source control.
- Add these entries to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
chroma_db/
```

The `chroma_db/` folder can be large and may contain text extracted from websites.

## Possible Improvements

- Crawl multiple pages from a domain.
- Add source links and citations to answers.
- Stream responses as they are generated.
- Use persistent LangGraph checkpoint storage for memory across restarts.
- Add a refresh/re-index option for changed webpages.
- Add tests for URL loading, retrieval, and agent responses.

## License

Choose and add a license before distributing this project publicly.
