import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables (hides API key)
# hehe
load_dotenv()

# Constants
CHROMA_PATH = "chroma_db"

# --- Page Configuration ---
st.set_page_config(page_title="MARVEL RAG", page_icon="🦸‍♂️", layout="centered")

# --- Custom Marvel CSS ---
st.markdown("""
<style>
    /* Marvel-style Title */
    h1 {
        background-color: #EC1D24; /* Official Marvel Red */
        color: #FFFFFF !important;
        text-align: center;
        font-family: 'Impact', 'Arial Black', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        margin-bottom: 0px;
    }

    /* Subtitle text */
    .subtitle {
        text-align: center; 
        font-weight: bold; 
        color: #555;
        font-size: 18px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("MARVEL KNOWLEDGE BASE")
st.markdown("<div class='subtitle'>Ask me anything about the Marvel Universe!</div>", unsafe_allow_html=True)
st.divider()


# --- Load Models (Cached for speed) ---
@st.cache_resource
def load_rag_pipeline():
    """Loads the embedding model, vector DB, and LLM just once to save time."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    # Using the fast Llama 3.1 model via Groq
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
    return vector_db, llm


vector_db, llm = load_rag_pipeline()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Processing ---
if prompt := st.chat_input("Ask a question about Marvel..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Accessing S.H.I.E.L.D. Database..."):
            # Retrieve relevant chunks from ChromaDB
            results = vector_db.similarity_search(prompt, k=5)

            # Format the context text
            context_text = ""
            for doc in results:
                source = doc.metadata.get('title', 'Unknown Source')
                context_text += f"\n[Source: {source}]\n{doc.page_content}\n"

            # Define the strict System Prompt
            prompt_template = """You are a helpful and precise assistant. Your task is to answer the user's question based strictly on the provided context documents.

            CRITICAL INSTRUCTIONS:
            1. Answer ONLY using the information from the provided context.
            2. You MUST cite the source document by name for every factual claim you make (e.g., "According to [Source Name], ...").
            3. If the answer is not contained in the provided documents, you MUST respond exactly with this phrase: "I cannot find this in the provided documents." Do not invent or guess anything.

            Context Documents:
            {context}

            User Question: {question}

            Answer:"""

            qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = qa_prompt | llm

            # Generate response
            response = chain.invoke({"context": context_text, "question": prompt})
            answer = response.content

            # Display the answer
            st.markdown(answer)

            # Add an expander to let the user see the retrieved context (Great for debugging/demo!)
            with st.expander("🔍 View Retrieved Context"):
                st.text(context_text)

    # 3. Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})