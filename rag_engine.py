import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. Load environment variables (API Key)
load_dotenv()

class RAGService:
    """
    Service class handling Retrieval-Augmented Generation (RAG) operations.
    Responsible for document ingestion, text chunking, vector storage, and query execution.
    """
    def __init__(self):
        # Initialize the Embeddings model (converts text into dense vector representations)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Initialize the LLM (the reasoning engine)
        self.llm = ChatOpenAI(model_name="gpt-5-nano", reasoning={"effort": "minimal"})
        
        self.vector_store = None
        
        # Implement conversation memory to maintain context across multiple queries
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def process_pdf(self, pdf_path: str) -> bool:
        """
        Simulates the data ingestion (ETL) pipeline.
        Reads a PDF, splits it into manageable chunks, and creates a vector database.
        
        Args:
            pdf_path (str): The file path to the PDF document.
            
        Returns:
            bool: True if processing is successful.
        """
        print(f"--- Processing document: {pdf_path} ---")
        
        # A. LOAD: Extract text from the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # B. SPLIT: Divide text into chunks
        # This step is crucial to mitigate LLM Context Window limitations and improve retrieval accuracy.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,   # Maximum characters per chunk
            chunk_overlap=200  # Overlap to prevent breaking sentences/context in half
        )
        chunks = text_splitter.split_documents(documents)
        print(f"--- Document successfully split into {len(chunks)} chunks ---")

        # C. EMBED & STORE: Vectorize chunks and store them in FAISS
        # FAISS serves as an efficient, local, in-memory vector database
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("--- Vector database initialized successfully ---")
        
        return True

    def get_qa_chain(self) -> ConversationalRetrievalChain:
        """
        Constructs the conversational retrieval chain linking the LLM, Memory, and Vector Store.
        
        Returns:
            ConversationalRetrievalChain: The configured LangChain QA chain.
        """
        if not self.vector_store:
            raise ValueError("You must process a PDF document before initializing the QA chain.")

        # Configure the retriever to fetch the top 3 most semantically similar chunks
        retriever = self.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3} 
        )

        # Assemble the RAG chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory
        )
        
        return qa_chain

# --- TEST BLOCK ---
# This block demonstrates basic manual unit testing and only runs if the script is executed directly.
if __name__ == "__main__":
    try:
        # Initialize the RAG engine
        engine = RAGService()
        
        # Simulate file upload and ingestion pipeline
        engine.process_pdf("The-Complete-Guide-to-Building-Skill-for-Claude.pdf")
        
        # Retrieve the configured QA chain
        qa = engine.get_qa_chain()
        
        # Define test queries to validate information extraction and context awareness
        questions = [
            "What is the main topic of this document?", 
            "Can you summarize the key takeaways?"
        ]
        
        for q in questions:
            print(f"\nQuestion: {q}")
            response = qa.invoke({"question": q})
            print(f"AI Response: {response['answer']}")
            
    except Exception as e:
        print(f"Execution Error: {e}")