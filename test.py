import sys
import os

print(f"Python Path: {sys.executable}")
try:
    import langchain
    print(f"Langchain location: {langchain.__file__}")
    from langchain.chains import ConversationalRetrievalChain
    print("Import successful!")
except Exception as e:
    print(f"Failed: {e}")