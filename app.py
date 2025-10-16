# coding: utf-8
"""
Implementa uma aplicação de chatbot usando Streamlit e LangChain com o modelo DeepSeek.
"""

import streamlit as st
from langchain_ollama import ChatOllama

MODEL_NAME = "deepseek-r1:1.5b"


def main() -> None:
    """
    Função principal para executar a aplicação Streamlit com ChatOllama.
    Configura o modelo, inicializa o estado da sessão e gerencia a interação do usuário
    com o chatbot.
    Returns:
        None
    """
    
    # Configurações do ChatOllama e Streamlit
    chat_ollama = ChatOllama(model=MODEL_NAME, base_url="http://localhost:11434")
    st.set_page_config(page_title="Chat Ollama DeepSeek", layout="centered")
    st.title("Tudo pronto? Então vamos lá!")

    # Inicializar estado da sessão para armazenar mensagens
    if "mensagens" not in st.session_state:
        st.session_state["mensagens"] = []
    mensagens = st.session_state["mensagens"]

    # Exibir mensagens anteriores
    for tipo, conteudo in mensagens:
        chat = st.chat_message(tipo)
        chat.markdown(conteudo)

    # Entrada do usuário
    prompt = st.chat_input("Pergunte alguma coisa")

    if prompt:
        
        # Adicionar pergunta do usuário ao histórico de mensagens
        mensagens.append(("human", prompt))

        # Exibir pergunta do usuário
        chat = st.chat_message("human")
        chat.markdown(prompt)

        # Obter resposta da IA e adicionar ao histórico de mensagens
        resposta = chat_ollama.invoke(mensagens).content
        mensagens.append(("ai", resposta))

        # Exibir resposta da IA
        chat = st.chat_message("ai")
        chat.markdown(resposta)


if __name__ == "__main__":
    main()
