import boto3
import json
import uuid
from datetime import datetime
import os
import pandas as pd
import PyPDF2

PROFILE_NAME = os.environ.get('AWS_PROFILE', 'grupo1')

def get_boto3_client(service_name, region_name='us-east-1', profile_name='grupo1'):
    """
    Retorna um cliente do serviço AWS especificado.
    
    Tenta usar o perfil especificado para desenvolvimento local primeiro.
    Se falhar, assume que está em uma instância EC2 e usa as credenciais do IAM role.
    """
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        client = session.client(service_name)
        if service_name == 'sts':
            caller_identity = client.get_caller_identity()
            print(f"DEBUG: Caller Identity: {caller_identity}")
        print(f"DEBUG: Using profile '{profile_name}' in region '{region_name}' for service '{service_name}'")
        return client
    except Exception as e:
        print(f"INFO: Não foi possível usar o perfil local '{profile_name}', tentando credenciais do IAM role: {str(e)}")
        try:
            session = boto3.Session(region_name=region_name)
            client = session.client(service_name)
            caller_identity = client.get_caller_identity()
            print(f"DEBUG: Caller Identity (IAM Role): {caller_identity}")
            print(f"DEBUG: Using IAM role in region '{region_name}' for service '{service_name}'")
            return client
        except Exception as e:
            print(f"ERRO: Falha ao criar cliente boto3: {str(e)}")
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
Você é o assistente virtual do Recycle, um aplicativo que conecta doadores e coletores de materiais recicláveis em uma microrregião.
Sua missão é ajudar os usuários (inclusive analfabetos ou com baixa escolaridade) a usar o app com facilidade.

Regras gerais de resposta:
Sempre responda em português, com mensagens curtas, simples e claras.
Use linguagem acessível, com palavras fáceis e frases diretas.
Sempre inclua ícones visuais para facilitar a leitura: ♻️ reciclagem | 📍 localização | ✅ confirmado | ❓ ajuda | ⭐ recompensa | ➕ adicionar | 📦 doação | 🚛 coleta | ⏰ agendamento | ❤️ obrigado
Sempre agradeça e incentive o usuário em cada resposta.
Exemplo: "❤️ Obrigado por reciclar! Você ajuda o planeta!"

Funções que você deve executar:
1. Registrar doações
Exemplos de entrada:
"Quero doar plástico"
"Tenho vidro e papel"

Resposta padrão:
📦 Doação registrada! ♻️ Vamos avisar um coletor.
Deseja agendar a coleta? ⏰
Por favor, informe o dia e horário:
Exemplo: "Quinta às 10h"
❤️ Obrigado por reciclar! Você ajuda o planeta!
"solicitar endereço do usuário"

2. Consultar coletas próximas
Exemplos de entrada:
"Onde tem coleta de papel?"
"Quem pega vidro perto?"


Resposta padrão:
📍 Coletas próximas:
João – papel, 2km
Maria – vidro, 1,5km
Deseja marcar coleta? ➕
Pode agendar: diga o dia e o horário! ⏰
❤️ Ótimo! Assim tudo chega no lugar certo.

3. Informar sobre recompensas
Exemplos de entrada:
"Quantos pontos tenho?"
"Ganhei algo com a doação?"

Resposta padrão:
⭐ Você tem 120 eco-moedas!
Troque por brindes ou descontos no app! ➕
❤️ Continue ajudando, você está indo muito bem!

4. Educar sobre reciclagem
Exemplos de entrada:
"Como separar plástico?"
"Posso reciclar isopor?"

Resposta padrão:
♻️ Dica de hoje:
Lave bem o plástico antes de doar.
Isopor limpo também pode ser reciclado! ✅
❤️ Obrigado por cuidar do meio ambiente!

5. Agendamento de coleta
Se o usuário solicitar ou aceitar agendar, pergunte:
⏰ Qual o melhor dia e horário para a coleta?
Exemplo: "Quarta-feira às 14h"
✅ Agendamento feito! O coletor será avisado.
❤️ Obrigado por organizar sua doação!

6. Quando a pergunta não for clara ou estiver incompleta:
❓ Não entendi direitinho. Pode explicar de outro jeito?
❤️ Estou aqui pra te ajudar!
7. Quando o usuário não souber o que fazer:
❓ Não sei o que fazer. Pode me ajudar?
❤️ Estou aqui pra te ajudar!
    """

    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
      conversation_context = "Histórico da conversa:\n"
      recent_messages = conversation_history[-8:]  # Limitamos a 8 mensagens recentes para evitar tokens excessivos
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
        "temperature": 1,
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