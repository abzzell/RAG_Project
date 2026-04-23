import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables from .env file (to hide our API key)
load_dotenv()

# We need the path to our database
CHROMA_PATH = "chroma_db"


def generate_answer(query):
    # 1. Initialize the LLM (using Groq's fast Llama 3 model)
    # The API key is automatically picked up from the GROQ_API_KEY variable in .env
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)

    # 2. Initialize database and search for relevant chunks
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

    # Project requirement: Retrieve the top-5 most similar chunks
    results = vector_db.similarity_search(query, k=5)

    # 3. Format the context and extract source titles
    context_text = ""
    for doc in results:
        source = doc.metadata.get('title', 'Unknown Source')
        context_text += f"\n[Source: {source}]\n{doc.page_content}\n"

    # 4. Define the System Prompt strictly following project requirements
    prompt_template = """You are a helpful and precise assistant. Your task is to answer the user's question based strictly on the provided context documents.

    CRITICAL INSTRUCTIONS:
    1. Answer ONLY using the information from the provided context.
    2. You MUST cite the source document by name for every factual claim you make (e.g., "According to [Source Name], ...").
    3. If the answer is not contained in the provided documents, you MUST respond exactly with this phrase: "I cannot find this in the provided documents." Do not invent or guess anything.

    Context Documents:
    {context}

    User Question: {question}

    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 5. Generate the response
    chain = prompt | llm

    print(f"\n--- Question: '{query}' ---")
    print("Thinking and generating...\n")



    response = chain.invoke({"context": context_text, "question": query})

    response = chain.invoke({"context": context_text, "question": query})

    print("--- Final Answer ---")
    print(response.content)
    print("-" * 50)


if __name__ == "__main__":
    # Test 1: A question that can be answered from our documents
    generate_answer("Who is in the Avengers roster?")

    # Test 2: A question that triggers the refusal behavior (Out-of-context)
    generate_answer("How do I bake a chocolate cake?")