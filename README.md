# RAG Question Answering API

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, FAISS, Sentence-Transformers, and OpenAI. 

This API allows users to upload documents (PDF, TXT, DOCX, MD) and ask natural language questions about them. The system intelligently searches the documents and uses an LLM to generate precise answers based *only* on the provided context.

---

## 🧠 How It Works

The system operates in two main phases: **Ingestion** (reading the document) and **Retrieval/Generation** (answering the question).

### Flow Diagram
```text
[ Document ] ──> Parse ──> Chunk ──> Embed ──> [ FAISS Vector Store ]
                                                       │
[ Question ] ──> Embed ────(Similarity Search)─────────┘
                               │
                       [ Top 5 Chunks ] ──+── [ Prompt ] ──> [ OpenAI LLM ] ──> [ Answer ]
Step-by-Step Explanation
Phase 1: Document Ingestion (Background Job)
Upload: You upload a document via the /documents endpoint. The file is saved to the local disk, and a background task is triggered so the API doesn't freeze.
Parsing: The system reads the raw text from the file (using pdfplumber for PDFs, python-docx for Word docs, etc.).
Chunking: The text is split into small, manageable pieces (512 characters each). It uses a "sliding window" approach with a 64-character overlap so that sentences aren't abruptly cut in half.
Embedding: Each chunk is converted into a mathematical vector (an array of 384 numbers) using the local all-MiniLM-L6-v2 model. This allows the system to understand the meaning of the text.
Storage: These vectors, along with the original text, are saved into a local FAISS vector database.
Phase 2: Querying
Ask: You submit a question via the /query endpoint.
Query Embedding: The system converts your question into a vector using the exact same local model.
Similarity Search: The system compares the question's vector against all document vectors in the FAISS database to find the top 5 most mathematically similar (relevant) chunks.
Generation: The system sends those 5 chunks of text, along with your original question, to OpenAI (gpt-3.5-turbo). The AI is strictly instructed to answer only using the provided chunks.
Response: The API returns the generated answer, the exact text chunks it used as sources, and the latency time.
🚀 Features
Multi-format Support: Accepts .pdf, .txt, .docx, and .md files.
Asynchronous Processing: Fast uploads with background document processing.
Local Embeddings: Uses free, fast, local HuggingFace embeddings (sentence-transformers) to save on API costs.
Advanced Chunking: Character-level sliding window chunker with sentence-boundary awareness.
Rate Limiting: Built-in API rate limiting using slowapi to prevent spam.
Validation: Strict request validation using Pydantic.
💻 Tech Stack
Web Framework: FastAPI
Vector Store: FAISS (Facebook AI Similarity Search)
Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)
LLM: OpenAI (gpt-3.5-turbo)
Python Libraries: Pydantic, python-dotenv, pdfplumber, python-docx, slowapi
🛠️ Setup & Installation
1. Prerequisites
Python 3.10+ installed on your system.
An active OpenAI API Key.
2. Clone and Setup Environment
Open your terminal and run the following commands:
code
Bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# For Windows:
.venv\Scripts\activate
# For Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install python-dotenv
3. Environment Variables
Create a file named .env in the root directory of the project and add your OpenAI API key:
code
Env
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=5
MAX_FILE_SIZE_MB=20
🏃‍♂️ Running the Application
Start the FastAPI server using Uvicorn:
code
Bash
uvicorn app.main:app --reload --port 8000
The server will start at http://127.0.0.1:8000.
(Note: The first time you run the server, it will download the ~80MB embedding model automatically).
📖 Usage Instructions
The easiest way to interact with the API is through the built-in Swagger UI.
Open your browser and navigate to: http://localhost:8000/docs
Upload a Document:
Expand POST /api/v1/documents.
Click Try it out, choose a PDF or TXT file, and click Execute.
Check Status:
Expand GET /api/v1/documents/{document_id}.
Paste the ID you received from the upload step to ensure the status is "ready".
Ask a Question:
Expand POST /api/v1/query.
Click Try it out and edit the JSON body to ask your question:
code
JSON
{
  "question": "What is the main topic of the document?",
  "top_k": 5
}
Click Execute to get your AI-generated answer based on the document!
📊 Design Decisions & Metrics
Chunk Size (512 chars): Chosen to fit comfortably within the all-MiniLM-L6-v2 128-token context window, preventing silent truncation while remaining large enough to hold complete thoughts.
Similarity Thresholds: Cosine similarity is tracked via FAISS IndexFlatIP (with normalized vectors).
Latency Tracking: End-to-end LLM generation latency is tracked and returned in every query response to monitor system performance.



ALSO THE ENV file that has to be pressent in the same folder as app data and tests are. The content tof the .env file is

OPENAI_API_KEY=sk-(enter your OpenAI API key here)
OPENAI_MODEL=gpt-3.5-turbo
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=5
MAX_FILE_SIZE_MB=20
