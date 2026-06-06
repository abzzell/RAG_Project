# 🤖 Marvel Universe RAG Chatbot

This project is a Production-Ready RAG (Retrieval-Augmented Generation) pipeline designed to answer questions about the Marvel Universe based on specific PDF/Text documents. The system uses a vector database to retrieve relevant context and a Large Language Model (LLM) to generate grounded answers with source citations.

## 🛠 Tech Stack
- **LLM:** Llama 3.1-8B (via Groq API)
- **Vector DB:** ChromaDB
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (BERT-based)
- **Orchestration:** LangChain
- **UI:** Streamlit

## 📁 Project Structure
All project code is located in the root directory for straightforward execution:
- `ingest.py` / `vector_store.py`: Logic for PDF/TXT loading, text chunking, and creating the ChromaDB vector database.
- `generation.py`: Prompt engineering, LLM chain setup, and grounding logic.
- `app.py`: Streamlit web interface and semantic search retrieval logic.
- `evaluation.py`: Automated evaluation script for metrics (Precision@5).
- `eval_dataset.json` / `eval_results.json`: Datasets for evaluation and logging.


## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your Groq API Key:
```env
GROQ_API_KEY=your_api_key_here
```

### 3. Data Ingestion
Place your PDF/TXT files in the `data/` folder and run the ingestion script to create the vector database:
```bash
python3 vector_store.py
```

### 4. Running the Chatbot
Launch the Streamlit interface:
```bash
streamlit run app.py
```

### 5. Running Evaluation
To run the evaluation pipeline on 30+ QA pairs and compute metrics:
```bash
python3 evaluation.py
```

## 📊 Evaluation & Experiments
This system achieved a **Retrieval Precision@5 of 100%** on our test dataset of 30 questions. 
Detailed experiments with chunk sizes, k-parameters, and overlap are documented in the **Technical Report**.

## 🧠 Architecture Study
This project involves a comparative study of:
- **BERT (Encoder-only):** Used for creating semantic embeddings (Retrieval).
- **GPT-style (Decoder-only):** Used for generating natural language responses (Generation).
<img width="1271" height="843" alt="image" src="https://github.com/user-attachments/assets/9efe9dbb-e93a-4e57-82d7-ec5b867a8167" />
