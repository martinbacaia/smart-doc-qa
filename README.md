# 🧠 Smart Doc QA - RAG Backend Engine

A Retrieval-Augmented Generation (RAG) engine built in Python. It allows ingesting PDF documents, vectorizing them, and performing natural language queries while maintaining conversation context.

## 🚀 Tech Stack
* **Language:** Python 3.9+
* **Orchestrator:** LangChain
* **LLM & Embeddings:** OpenAI (gpt-5-nano)
* **Vector Database:** FAISS (In-memory search)

## ⚙️ Backend Architecture
The system implements *Separation of Concerns*. The `rag_engine.py` file handles document loading (`PyPDFLoader`), chunking to optimize the context window (`RecursiveCharacterTextSplitter`), and persistence in the vector database.

## 🛠️ Local Installation & Usage
1. Clone this repository: `git clone https://github.com/martinbacaia/smart-doc-qa.git`
2. Create a virtual environment: `python -m venv venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file in the root directory and add: `OPENAI_API_KEY=your_api_key_here`
5. Run the engine: `python rag_engine.py`