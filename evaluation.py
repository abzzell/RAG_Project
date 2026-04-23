import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()


def run_evaluation():
    CHROMA_PATH = "chroma_db"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    with open("eval_dataset.json", "r") as f:
        dataset = json.load(f)

    results_log = []
    correct_retrievals = 0

    print(f"Starting evaluation on {len(dataset)} questions...\n")

    for entry in dataset:
        query = entry["question"]
        expected_src = entry["context_source"]


        retrieved_docs = vector_db.similarity_search(query, k=5)

        found_sources = [doc.metadata.get('title', '') for doc in retrieved_docs]
        is_retrieval_correct = any(expected_src in src for src in found_sources) if expected_src != "None" else True

        if is_retrieval_correct:
            correct_retrievals += 1

        context_text = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt_template = "Answer the question based ONLY on context: {context}\nQuestion: {question}"
        prompt = PromptTemplate.from_template(prompt_template).format(context=context_text, question=query)

        response = llm.invoke(prompt)

        results_log.append({
            "question": query,
            "retrieval_success": is_retrieval_correct,
            "model_answer": response.content
        })
        print(f"Q: {query} | Retrieval: {'✅' if is_retrieval_correct else '❌'}")

    precision_at_5 = correct_retrievals / len(dataset)

    print(f"\n--- FINAL METRICS ---")
    print(f"Retrieval Precision@5: {precision_at_5:.2%}")
    print(f"Retrieval Precision@5: {precision_at_5:.2%}")
    print(f"Total Questions: {len(dataset)}")

    with open("eval_results.json", "w") as f:
        json.dump(results_log, f, indent=4)


if __name__ == "__main__":
    run_evaluation()