import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from ingest import load_all_documents
from chunking import strategy_2_recursive

CHROMA_PATH = "chroma_db"


def create_vector_db():
    # 1. Load and chunk documents
    print("Loading and chunking documents...")
    docs = load_all_documents("./data")
    chunks = strategy_2_recursive(docs)

    # 2. Initialize the embedding model
    # Using a lightweight and fast model from HuggingFace
    print("Loading the embedding model (this may take a couple of minutes the first time)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Create the vector database and save it to disk
    print("Creating ChromaDB vector store...")

    # To let ChromaDB understand our dictionaries, we need to split them into lists of texts and metadata
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]

    # Create and persist the database
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH
    )

    print(f"Database successfully created and saved to {CHROMA_PATH}!")
    return vector_db


def test_search(query):
    """Tests the search functionality of the database"""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

    print(f"\nQuestion: '{query}'")
    print("Searching the database...\n")

    # Search for the top 3 most relevant chunks
    results = vector_db.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(results):
        print(f"--- Result {i + 1} (Similarity: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('title', 'Unknown')}")
        print(f"Text: {doc.page_content[:200]}...\n")


if __name__ == "__main__":
    # If the database doesn't exist yet, create it
    if not os.path.exists(CHROMA_PATH):
        create_vector_db()

    # Test the search
    test_search("What is Vibranium made of?")
    test_search("Who created the Infinity Stones?")