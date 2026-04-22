from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from ingest import load_all_documents  # Import the function from our first script


def strategy_1_fixed_size(documents, chunk_size=500, chunk_overlap=50):
    """
    Strategy 1: Naive splitting by a fixed number of characters.
    chunk_overlap ensures we don't lose context at the boundaries of the chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    for doc in documents:
        # Split the document's text
        chunks = text_splitter.split_text(doc['text'])

        # Save chunks along with their source document metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'metadata': doc['metadata'],
                'chunk_index': i,
                'strategy': 'fixed_size'
            })

    return all_chunks


def strategy_2_recursive(documents, chunk_size=500, chunk_overlap=50):
    """
    Strategy 2: Smart recursive splitting (tries to keep paragraphs and sentences intact).
    It looks for double newlines (\n\n) first, then single (\n), then spaces.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc['text'])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'metadata': doc['metadata'],
                'chunk_index': i,
                'strategy': 'recursive'
            })

    return all_chunks


if __name__ == "__main__":
    folder_path = "./data"

    print("Loading documents...")
    docs = load_all_documents(folder_path)

    print("\n--- Strategy 1 (Fixed Size) ---")
    chunks_fixed = strategy_1_fixed_size(docs)
    print(f"Total chunks created: {len(chunks_fixed)}")
    print(f"Chunk example: {chunks_fixed[0]['text'][:100]}...\n")

    print("--- Strategy 2 (Recursive) ---")
    chunks_recursive = strategy_2_recursive(docs)
    print(f"Total chunks created: {len(chunks_recursive)}")
    print(f"Chunk example: {chunks_recursive[0]['text'][:100]}...\n")