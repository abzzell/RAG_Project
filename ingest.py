import os
import PyPDF2


def process_txt_file(filepath):

    metadata = {}
    content_lines = []

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for line in lines:
            if line.startswith('Title:'):
                metadata['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Source:'):
                metadata['source'] = line.replace('Source:', '').strip()
            elif line.startswith('Date:'):
                metadata['date'] = line.replace('Date:', '').strip()
            else:
                if line.strip():
                    content_lines.append(line.strip())

    if 'title' not in metadata:
        metadata['title'] = os.path.basename(filepath)

    return {
        'metadata': metadata,
        'text': ' '.join(content_lines)
    }


def process_pdf_file(filepath):

    metadata = {'source': 'Wikipedia PDF'}
    metadata['title'] = os.path.basename(filepath).replace('.pdf', '')
    text_content = ""

    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text_content += extracted_text + " "

    return {
        'metadata': metadata,
        'text': text_content.strip()
    }


def load_all_documents(data_folder):

    all_documents = []

    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)

        if filename.endswith('.txt'):
            print(f"Loading TXT: {filename}")
            doc = process_txt_file(filepath)
            all_documents.append(doc)

        elif filename.endswith('.pdf'):
            print(f"Loading PDF: {filename}")
            doc = process_pdf_file(filepath)
            all_documents.append(doc)

    return all_documents


if __name__ == "__main__":
    folder_path = "./data"

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found. Create it and put files there.")
    else:
        documents = load_all_documents(folder_path)
        print(f"\nSuccessfully loaded documents: {len(documents)}")

        if documents:
            print("\n--- First document preview ---")
            print(f"Metadata: {documents[0]['metadata']}")
            print(f"Text start: {documents[0]['text'][:200]}...")