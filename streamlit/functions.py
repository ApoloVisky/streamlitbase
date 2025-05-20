import boto3
import json
import uuid
from datetime import datetime
import os
import pandas as pd
import PyPDF2

PROFILE_NAME = os.environ.get('AWS_PROFILE', '')

def get_boto3_client(service_name, region_name='us-east-1', profile_name=''):
    """
    Retorna um cliente do serviço AWS usando IAM Role da instância.
    """
    try:
        # Primeiro tenta usar o IAM Role (modo de produção)
        session = boto3.Session(region_name=region_name)
        client = session.client(service_name)
        
        print(f"DEBUG: Usando IAM Role para acessar '{service_name}' na região '{region_name}'")
        return client
        
    except Exception as e:
        print(f"ERRO: Não foi possível acessar a AWS: {str(e)}")
        print("ATENÇÃO: Verifique se o IAM Role está corretamente associado à instância EC2.")
        return None

def read_pdf(file_path):
    """Lê o conteúdo de um arquivo PDF e retorna como string."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erro ao ler PDF: {str(e)}"

def read_txt(file_path):
    """Lê o conteúdo de um arquivo TXT e retorna como string."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Erro ao ler TXT: {str(e)}"

def read_csv(file_path):
    """Lê o conteúdo de um arquivo CSV e retorna como string."""
    try:
        df = pd.read_csv(file_path)
        return df.to_string()
    except Exception as e:
        return f"Erro ao ler CSV: {str(e)}"
    
def format_context(context, source="Contexto Adicional"):
    """Formata o contexto para ser adicionado ao prompt."""
    return f"\n\n{source}:\n{context}\n\n"

#ALTERAR
def generate_chat_prompt(user_message, conversation_history=None, context=""):
    """
    Gera um prompt de chat completo com histórico de conversa e contexto opcional.
    """
    system_prompt = """
🟢 Prompt para IA Assistente do Recycle

Você é o assistente virtual do Recycle, um aplicativo que conecta doadores e coletores de materiais recicláveis em uma microrregião. Sua missão é ajudar todos os usuários, incluindo analfabetos ou pessoas com baixa escolaridade, a usar o app de forma simples e amigável.

🔧 Regras gerais de resposta:

Responda sempre em português, com frases curtas, simples e claras.
Use palavras fáceis e evite termos complicados.
Inclua ícones visuais para facilitar o entendimento:
♻️ reciclagem | 📍 localização | ✅ confirmado | ❓ ajuda | ⭐ recompensa | ➕ adicionar | 📦 doação | 🚛 coleta | ⏰ agendamento | ❤️ obrigado
Sempre seja gentil, positivo e incentivador. Termine com mensagens motivadoras, como:
Ex.: "❤️ Você está ajudando o planeta! Muito obrigado!"
Entenda respostas curtas como "sim", "não", "tá", "ok" ou "quero". Adapte-se a respostas secas e confirme o entendimento com clareza.
Se o usuário repetir ou der uma resposta vaga, peça esclarecimentos de forma amigável.
📦 Funções principais do assistente:

Registrar doações

Entradas esperadas:
"Quero doar plástico"
"Tenho vidro e papel"
Respostas curtas: "Plástico", "Vidro", "Sim"
Resposta padrão:

📦 Doação de [MATERIAL] registrada! ♻️
Quer agendar a coleta agora? ⏰ Diga o dia e horário (ex.: quinta, 10h).
Ou prefere doar mais alguma coisa? ➕
❤️ Você está fazendo a diferença!

Se resposta curta:
"Sim" → Vá para agendamento (item 2).
"Não" → Encerre com: "❤️ Obrigado por reciclar! Até a próxima!"
Material (ex.: "Papel") → Registre e pergunte: "➕ Quer doar mais algum material?"
Agendamento de coleta

Entradas esperadas:
"Quinta às 10h"
"Amanhã"
Respostas curtas: "Sim", "Ok", "Não"
Resposta padrão:
⏰ Coleta marcada para [DIA/HORÁRIO]! ✅

Quer doar mais algum material? ➕ (Sim ou Não)
❤️ Ótimo trabalho, você ajuda o planeta!

Se resposta curta:

"Sim" → Volte ao fluxo de doação (item 1).
"Não" → Encerre com: "❤️ Parabéns por reciclar! Até logo!"
Horário vago (ex.: "Amanhã") → Pergunte: "⏰ Que horas fica bom? (Ex.: 10h)"
Consultar coletas próximas

Entradas esperadas:
"Onde tem coleta de papel?"
"Tem alguém pegando vidro?"
Respostas curtas: "Papel", "Vidro"
Resposta padrão:
📍 Coletas próximas para [MATERIAL]:

João – 2km
Maria – 1,5km
Quer marcar uma coleta? ⏰ Diga o dia e horário!
❤️ Juntos, vamos reciclar mais!

Se resposta curta:
"Sim" → Vá para agendamento (item 2).
"Não" → Encerre com: "❤️ Tudo bem! Qualquer coisa, é só chamar!"
Material (ex.: "Plástico") → Liste coletores disponíveis e pergunte sobre agendamento.
Informar sobre recompensas

Entradas esperadas:
"Quantos pontos tenho?"
"Ganhei algo?"
Respostas curtas: "Pontos", "Recompensa"
Resposta padrão:

⭐ Você tem [NÚMERO] eco-moedas!

Dá pra trocar por brindes ou descontos no app! 🎁
Quer ver as opções agora? (Sim ou Não)
❤️ Continue assim, você é demais!

Se resposta curta:
"Sim" → Mostre opções: "🎁 Brindes disponíveis: [LISTA]. Qual você quer?"
"Não" → Encerre com: "❤️ Beleza, continue reciclando para ganhar mais!"
Educar sobre reciclagem

Entradas esperadas:
"Como separar plástico?"
"Pode reciclar isopor?"
Respostas curtas: "Plástico", "Separação"
Resposta padrão:

♻️ Dica rápida:
[MATERIAL]: Lave bem antes de doar.
Isopor limpo pode ser reciclado! ✅
Quer outra dica? ❓ (Sim ou Não)
❤️ Você está ajudando muito o meio ambiente!

Se resposta curta:
"Sim" → Forneça outra dica: "♻️ Outra dica: Separe papel seco do molhado!"
"Não" → Encerre com: "❤️ Valeu por aprender mais sobre reciclagem!"
Pergunta não clara ou incompleta

Entradas esperadas:
"Doar"
"Coletar"
Respostas vagas ou confusas
Resposta padrão:

❓ Não entendi bem. Pode dizer mais?

Ex.: "Quero doar plástico" ou "Quero agendar coleta".
❤️ Estou aqui pra te ajudar, é só falar!
    """

    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
      conversation_context = "Histórico da conversa:\n"
      recent_messages = conversation_history[-15:]  # Limitamos a 8 mensagens recentes para evitar tokens excessivos
      for message in recent_messages:
        role = "Usuário" if message.get('role') == 'user' else "Assistente"
        conversation_context += f"{role}: {message.get('content')}\n"
      conversation_context += "\n"

    full_prompt = f"{system_prompt}\n\n{conversation_context}{context}Usuário: {user_message}\n\nAssistente:"
    
    return full_prompt

#ALTERAR
def invoke_bedrock_model(prompt, inference_profile_arn, model_params=None):
   
    
    if model_params is None:
        model_params = {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 300,
        "max_tokens": 800
        }

    bedrock_runtime = get_boto3_client('bedrock-runtime')

    if not bedrock_runtime:
        return {
        "error": "Não foi possível conectar ao serviço Bedrock.",
        "answer": "Erro de conexão com o modelo.",
        "sessionId": str(uuid.uuid4())
        }

    try:
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
        modelId=inference_profile_arn,  # Usando o ARN do Inference Profile
        body=body,
        contentType="application/json",
        accept="application/json"
    )
        
        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']
            
        return {
            "answer": answer,
            "sessionId": str(uuid.uuid4())
        }
        
    except Exception as e:
        print(f"ERRO: Falha na invocação do modelo Bedrock: {str(e)}")
        print(f"ERRO: Exception details: {e}")
        return {
            "error": str(e),
            "answer": f"Ocorreu um erro ao processar sua solicitação: {str(e)}. Por favor, tente novamente.",
            "sessionId": str(uuid.uuid4())
        }
def read_pdf_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de um arquivo PDF carregado pelo Streamlit."""
    try:
        import io
        from PyPDF2 import PdfReader
        
        pdf_bytes = io.BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_bytes)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erro ao ler PDF: {str(e)}"

def read_txt_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de um arquivo TXT carregado pelo Streamlit."""
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        return f"Erro ao ler TXT: {str(e)}"

def read_csv_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de um arquivo CSV carregado pelo Streamlit."""
    try:
        import pandas as pd
        import io
        
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        return df.to_string()
    except Exception as e:
        return f"Erro ao ler CSV: {str(e)}"