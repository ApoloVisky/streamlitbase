import streamlit as st
import requests
import json
import hmac
import uuid
import time
import boto3
from datetime import datetime
import re
import base64
import os
from functions import (
    generate_chat_prompt, format_context, 
    read_pdf_from_uploaded_file, read_txt_from_uploaded_file, read_csv_from_uploaded_file
)

# Configurações iniciais
PROFILE_NAME = os.environ.get("AWS_PROFILE", "grupo1")
INFERENCE_PROFILE_ARN = "arn:aws:bedrock:us-east-1:851614451056:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# Inicialização da sessão
if 'password_correct' not in st.session_state:
    st.session_state.password_correct = False

st.set_page_config(
    page_title="Recycle",
    page_icon="logo.jpeg",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo_path = "logo.jpeg"

def add_javascript():
    """Adiciona JavaScript para melhorar a interação do usuário com o chat"""
    js_code = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const textarea = document.querySelector('textarea');
            if (textarea) {
                textarea.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const sendButton = document.querySelector('button[data-testid="baseButton-secondary"]');
                        if (sendButton) {
                            sendButton.click();
                        }
                    }
                });
            }
        }, 1000);
    });
    </script>
    """
    st.components.v1.html(js_code, height=0)

def check_password():
    """Verifica se o usuário está autenticado"""
    if st.session_state.get("password_correct", False):
        return True
        
    with st.sidebar:
        st.write("## Autenticação")
        password = st.text_input("Senha", type="password")
        
        if st.button("Login"):
            # Aqui você pode implementar sua lógica de verificação de senha
            # Exemplo simples (substitua por sua lógica real):
            if password == "admin123":  # Senha temporária para teste
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("Senha incorreta")
    
    return False

def get_boto3_client(service_name, region_name='us-east-1', profile_name='grupo1'):
    """Retorna um cliente do serviço AWS especificado"""
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        client = session.client(service_name)
        return client
    except Exception as e:
        print(f"Erro ao criar cliente boto3: {str(e)}")
        try:
            session = boto3.Session(region_name=region_name)
            return session.client(service_name)
        except Exception as e:
            print(f"Falha ao usar IAM role: {str(e)}")
            return None

def query_bedrock(message, session_id="", model_params=None, context="", conversation_history=None):
    """Envia uma mensagem para o Amazon Bedrock"""
    if model_params is None:
        model_params = {
            "temperature": 1.0,
            "top_p": 0.85,
            "top_k": 200,
            "max_tokens": 800,
            "response_format": {"type": "text"}
        }
    
    bedrock_runtime = get_boto3_client('bedrock-runtime')
    
    if not bedrock_runtime:
        return {
            "answer": "Não foi possível conectar ao serviço Bedrock. Verifique suas credenciais.",
            "sessionId": session_id or str(uuid.uuid4())
        }
    
    try:
        prompt = generate_chat_prompt(message, conversation_history=conversation_history, context=context)
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": model_params["max_tokens"],
            "temperature": model_params["temperature"],
            "top_p": model_params["top_p"],
            "top_k": model_params["top_k"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=INFERENCE_PROFILE_ARN,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']
        
        return {
            "answer": answer,
            "sessionId": session_id or str(uuid.uuid4())
        }
        
    except Exception as e:
        print(f"ERRO: Falha na requisição ao Bedrock: {str(e)}")
        return {
            "answer": "Ocorreu um erro ao processar sua solicitação. Por favor, tente novamente.",
            "sessionId": session_id or str(uuid.uuid4())
        }

def initialize_session():
    """Inicializa as variáveis de sessão necessárias"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = ""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_chat_index' not in st.session_state:
        st.session_state.current_chat_index = 0
    if 'chat_title' not in st.session_state:
        st.session_state.chat_title = f"Nova Conversa ({datetime.now().strftime('%d/%m/%Y')})"
    if 'use_rag' not in st.session_state:
        st.session_state.use_rag = False
    if 'file_to_send' not in st.session_state:
        st.session_state.file_to_send = None

    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "id": "",
            "title": st.session_state.chat_title,
            "messages": []
        })

def get_rag_context():
    """Obtém e formata o contexto RAG"""
    if st.session_state.get('use_rag', False):
        if st.session_state.rag_source == "Arquivo" and st.session_state.uploaded_file:
            file = st.session_state.uploaded_file
            if st.session_state.file_type == "PDF":
                return format_context(read_pdf_from_uploaded_file(file), f"Contexto do arquivo PDF: {file.name}")
            elif st.session_state.file_type == "TXT":
                return format_context(read_txt_from_uploaded_file(file), f"Contexto do arquivo TXT: {file.name}")
            elif st.session_state.file_type == "CSV":
                return format_context(read_csv_from_uploaded_file(file), f"Contexto do arquivo CSV: {file.name}")
        elif st.session_state.rag_source == "Texto Direto" and st.session_state.direct_text:
            return format_context(st.session_state.direct_text, "Contexto do Usuário")
    return ""

def handle_message():
    """Processa o envio de uma mensagem do usuário"""
    if st.session_state.user_input.strip():
        user_message = st.session_state.user_input.strip()
        timestamp = datetime.now().strftime("%H:%M")
        
        st.session_state.messages.append({"role": "user", "content": user_message, "time": timestamp})
        
        with st.chat_message("assistant", avatar=logo_path):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("_Digitando..._")
            
            with st.spinner():
                rag_context = get_rag_context()
                result = query_bedrock(
                    user_message,
                    st.session_state.session_id,
                    context=rag_context,
                    conversation_history=st.session_state.messages
                )
            
            if result:
                assistant_message = result['answer']
                st.session_state.session_id = result['sessionId']
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": assistant_message, 
                    "time": datetime.now().strftime("%H:%M")
                })
            
            typing_placeholder.empty()
        
        st.rerun()

def create_new_chat():
    """Cria uma nova conversa"""
    st.session_state.session_id = ""
    st.session_state.messages = []
    st.session_state.chat_title = f"Nova Conversa ({datetime.now().strftime('%d/%m/%Y')})"
    st.session_state.chat_history.append({
        "id": "",
        "title": st.session_state.chat_title,
        "messages": []
    })
    st.session_state.current_chat_index = len(st.session_state.chat_history) - 1
    st.rerun()

def load_chat(index):
    """Carrega uma conversa existente"""
    st.session_state.current_chat_index = index
    chat = st.session_state.chat_history[index]
    st.session_state.session_id = chat["id"]
    st.session_state.messages = chat["messages"].copy()
    st.session_state.chat_title = chat["title"]
    st.rerun()

def delete_chat(index):
    """Exclui uma conversa"""
    if len(st.session_state.chat_history) > index:
        st.session_state.chat_history.pop(index)
        if not st.session_state.chat_history:
            create_new_chat()
        elif st.session_state.current_chat_index >= len(st.session_state.chat_history):
            st.session_state.current_chat_index = len(st.session_state.chat_history) - 1
            load_chat(st.session_state.current_chat_index)
        st.rerun()

def logout():
    """Faz logout do usuário"""
    st.session_state.password_correct = False
    st.rerun()

# Estilos CSS
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
    }
    .message-time {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e6e6e6;
        z-index: 998;
    }
    .stButton button {
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Interface principal
if check_password():
    initialize_session()
    
    with st.sidebar:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(logo_path, width=50)
        with col2:
            st.markdown('<h2 style="margin-top: 0;">Chat IA</h2>', unsafe_allow_html=True)
        
        st.divider()
        st.button("🔄 Nova Conversa", on_click=create_new_chat, use_container_width=True)
        st.divider()
        
        st.markdown("### Minhas Conversas")
        for idx, chat in enumerate(st.session_state.chat_history):
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"📝 {chat['title']}", key=f"chat_{idx}", use_container_width=True):
                    load_chat(idx)
            with col2:
                if st.button("🗑️", key=f"delete_{idx}"):
                    delete_chat(idx)
        
        st.divider()
        if st.button("Logout", use_container_width=True):
            logout()

    # Área principal do chat
    main_col1, main_col2, main_col3 = st.columns([1, 10, 1])
    
    with main_col2:
        add_javascript()
        st.markdown(f'<div style="text-align: center; font-size: 1.5rem; margin-bottom: 1rem;">{st.session_state.chat_title}</div>', unsafe_allow_html=True)
        
        # Exibir mensagens
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
                    st.markdown(f"<div class='message-time'>{message['time']}</div>", unsafe_allow_html=True)
            else:
                with st.chat_message("assistant", avatar=logo_path):
                    st.write(message["content"])
                    st.markdown(f"<div class='message-time'>{message['time']}</div>", unsafe_allow_html=True)
        
        # Área de entrada de mensagens
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        user_input = st.text_area("Mensagem", placeholder="Digite sua mensagem aqui...", key="user_input", height=70, label_visibility="collapsed")
        
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Enviar", key="send_button", use_container_width=True):
                if user_input.strip():
                    handle_message()
        
        st.markdown('</div>', unsafe_allow_html=True)