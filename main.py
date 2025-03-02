import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings

from functions import get_response, load_vector_store

import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')
API_URL = 'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1'

embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
vector_store = load_vector_store(embeddings, vector_store_path="faiss_index")

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="centered-title">❓❔ RAG System App ❔❓</h1>', unsafe_allow_html=True)


st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcm9kY2QwNTBndHljcDBkeDJtZXhvd2NvcmwxZGc4M3ExOWFhM3pxciZlcD12MV9naWZzX3NlYXJjaCZjdD1n/GRk3GLfzduq1NtfGt5/giphy.gif" width="250">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .centered-textarea .stTextArea {
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="centered-textarea">', unsafe_allow_html=True)
user_input = st.text_area("Задайте любой вопрос! 🤔")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Спросить! 🚀"):
    with st.spinner("Думаю... 💭"):
        response = get_response(user_input, vector_store, API_URL, API_TOKEN)
        st.success("Вот что я нашел! 🎉")
        st.write(response.strip())