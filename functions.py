import re
import requests
from langchain.vectorstores import FAISS

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')
API_URL = 'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1'

# Загрузка векторного хранилища
def load_vector_store(embeddings, vector_store_path="faiss_index"):
    vector_store = FAISS.load_local(
        folder_path=vector_store_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True 
    )
    return vector_store

def create_prompt(query, relevant_docs):
    '''Функция, создающая промпт для модели-генератора'''

    # Очистка документов от технических артефактов
    cleaned_docs = [
        re.sub(r'doc_\d+', '', doc.page_content).strip()
        for doc in relevant_docs
    ]
    docs_summary = "\n".join(cleaned_docs)[:2000]  # Ограничение контекста

    prompt = f"""
    <s>[INST] <<SYS>>
    Язык ответа: Русский.
    Ты ассистент, отвечающий исключительно на основе предоставленных данных на русском языке. Строго соблюдай правила:

    1. Если ответа нет в данных → "Информация не найдена"
    2. Только 1 предложение (50-100 символов) на русском языке.
    3. Запрещено:
      - Упоминать источники/документы
      - Технические термины (doc_123, @sys)
      - Маркированные списки
      - Любые разделы кроме ответа
      - Выдумывать ответы
      - Повторять вопрос
    <</SYS>>

    ### Контекст ###
    {docs_summary}

    ### Примеры ###
    Вопрос: Какая площадь России?
    Ответ: 17.1 млн км².

    Вопрос: Сколько сейчас времени?
    Ответ: Информация не найдена.

    ### Задача ###
    Вопрос: {query}
    Ответ: [/INST]</s>
    """
    return prompt

def get_response(query, vector_store, api_url, api_token):
    '''
    Функция для получения ответа от языковой модели через API Hugging Face с использованием релевантных
    документов из хранилища FAISS.

    Args:
        query (str): Запрос пользователя
        vector_store (FAISS): Векторное хранилище FAISS
        api_url (str): URL API Hugging Face Inference Endpoint
        api_token (str): Токен API Hugging Face

    Returns:
        str: Ответ, сгенерированный языковой моделью.

    Raises:
        ValueError: Если API возвращает ошибку.
    '''
    try:
        # Получаем релевантные документы
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)

        # Формируем промт
        prompt = create_prompt(query, relevant_docs)

        # Отправляем запрос к API
        headers = {"Authorization": f"Bearer {api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 2000,
                "temperature": 0.4,
                "num_return_sequences": 1,
            }
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Проверяем HTTP ошибки

        # Обрабатываем успешный ответ
        output = response.json()
        full_response = output[0]["generated_text"][len(prompt):]
        
        # Используем регулярные выражения для корректного разделения на предложения
        pattern = r'(?<![А-Я][а-я]\.)(?<!\d\.)(?<![0-9])(?<!г\.)(?<!ул\.)(?<!пр\.)(?<!д\.)(?<!стр\.)(?<!корп\.)(?<!к\.)(?<!им\.)(?<!т\.)(?<!п\.)(?<=[.!?])\s+'
        sentences = re.split(pattern, full_response)
        first_sentence = sentences[0].strip()
        
        # Добавляем точку, если её нет в конце
        if not first_sentence[-1] in '.!?':
            first_sentence += '.'
            
        return first_sentence

    except requests.exceptions.HTTPError as err:
        error_msg = f"HTTP error ({err.response.status_code}): {err.response.text}"
        if err.response.status_code == 500:
            print(f"Server error: {error_msg}")
            return "Сервер временно недоступен, попробуйте позже"
        raise ValueError(error_msg) from err

    except requests.exceptions.RequestException as err:
        error_msg = f"Request failed: {str(err)}"
        print(error_msg)
        return "Сервер временно недоступен, попробуйте позже"

    except (KeyError, IndexError) as err:
        error_msg = f"Invalid API response format: {str(err)}"
        print(error_msg)
        raise ValueError(error_msg) from err