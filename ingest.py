import os
import PyPDF2


def process_txt_file(filepath):
    """
    Читает TXT файл, извлекает метаданные из первых строк и возвращает текст.
    """
    metadata = {}
    content_lines = []

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for line in lines:
            # Ищем метаданные в начале файла
            if line.startswith('Title:'):
                metadata['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Source:'):
                metadata['source'] = line.replace('Source:', '').strip()
            elif line.startswith('Date:'):
                metadata['date'] = line.replace('Date:', '').strip()
            else:
                # Если это не метаданные, значит это основной текст
                if line.strip():  # Игнорируем пустые строки
                    content_lines.append(line.strip())

    # Если название не указано внутри, берем имя файла
    if 'title' not in metadata:
        metadata['title'] = os.path.basename(filepath)

    return {
        'metadata': metadata,
        'text': ' '.join(content_lines)
    }


def process_pdf_file(filepath):
    """
    Читает PDF файл, извлекает текст и базовые метаданные (название файла).
    """
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
    """
    Проходит по всей папке data/, определяет тип файла и применяет нужную функцию.
    """
    all_documents = []

    # Перебираем все файлы в папке
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)

        if filename.endswith('.txt'):
            print(f"Загрузка TXT: {filename}")
            doc = process_txt_file(filepath)
            all_documents.append(doc)

        elif filename.endswith('.pdf'):
            print(f"Загрузка PDF: {filename}")
            doc = process_pdf_file(filepath)
            all_documents.append(doc)

    return all_documents


# Блок для проверки работы скрипта
if __name__ == "__main__":
    folder_path = "./data"

    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print(f"Ошибка: Папка {folder_path} не найдена. Создай ее и положи туда файлы.")
    else:
        documents = load_all_documents(folder_path)
        print(f"\nУспешно загружено документов: {len(documents)}")

        # Выведем превью первого документа, чтобы проверить, что всё работает
        if documents:
            print("\n--- Превью первого документа ---")
            print(f"Метаданные: {documents[0]['metadata']}")
            print(f"Начало текста: {documents[0]['text'][:200]}...")