import logging
import os
import pathlib
import sys

import azure.functions as func
from flask import Flask, request, jsonify, url_for, redirect, current_app, abort
from google.oauth2.service_account import Credentials as sa_credentials
import datetime
import io
import json
import os
from collections import deque
import pickle
import time
import traceback
from urllib.parse import parse_qs
from urllib.parse import urlparse
from uuid import uuid4

import elastic_transport
import slack
from flask import Flask, request, jsonify, url_for, redirect, current_app, abort
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials as sa_credentials
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchEmbeddings, ElasticsearchStore, ElasticsearchChatMessageHistory
from pypdf import PdfReader
from slack.errors import SlackApiError
from slackeventsapi import SlackEventAdapter
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from chat_ai import ask_question
from elasticsearch_client import (
    elasticsearch_client,
)

VECTOR_STORE = ""
INDEX = os.getenv("ES_INDEX", "workplace-app-docs")

SCOPES = ['https://www.googleapis.com/auth/drive']

SLACK_TOKEN = os.getenv("SLACK_TOKEN")
SLACK_SIGNIN_SECRET = os.getenv("SLACK_SIGNIN_SECRET")

app = Flask(__name__)


client = slack.WebClient(token=SLACK_TOKEN)

BOT_ID = client.api_call("auth.test")['user_id']

slack_event_adapter = SlackEventAdapter(
    SLACK_SIGNIN_SECRET, '/slack/events', app)

session_state = {}
conversation_history = []

now = datetime.datetime.now()
session_id = ''


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Each request is redirected to the WSGI handler.
    """
    return func.WsgiMiddleware(app.wsgi_app).handle(req, context)


@slack_event_adapter.on("message")
def handle_message(payload):
    try:
        # Definir session_id con un valor inicial vacío
        global session_id
        event = payload.get("event")
        user = event.get("user")
        question = event.get("text")
        channel_id = event.get('channel', str(uuid4()))
        result = client.conversations_history(channel=channel_id)
        conversation_history = result["messages"]
        penum_chat_ts = datetime.datetime.fromtimestamp(
            float(conversation_history[1]['ts']))
        final_time = now - penum_chat_ts

        if BOT_ID == user:
            return "error", 500
        if conversation_history[0]['user'] == BOT_ID:
            return "error", 500
        

        # Verificar si hay una sesión en curso en el canal
        if channel_id in session_state and session_state[channel_id]:
            # Si la sesión está en progreso, ignorar esta solicitud.
            return

        # Marcar la sesión como en progreso
        session_state[channel_id] = True

        try:
            if final_time.total_seconds() > 360:
                client.chat_postMessage(
                    channel="#rag-pruebas", text="El chat se ha finalizado, realice una pregunta de nuevo.", concurrent=1)
                session_id = str(uuid4())
            else:
                # Verificar si el mensaje fue enviado por el bot para evitar bucles infinitos

                if question is None:
                    client.chat_postMessage(
                        channel="#rag-pruebas", text="Por favor, escribe una pregunta válida")
                else:

                    if question.lower() == "hola":
                        client.chat_postMessage(
                            channel="#rag-pruebas", text="Hola")
                    else:
                        if not session_id:
                            session_id = str(uuid4())
                        # Obtener la respuesta a partir de la pregunta
                        print(session_id)
                        respuesta = ask_question(question, session_id)
                        respuesta = "".join(list(respuesta))

                        # Publicar la respuesta en el canal de pruebas
                        client.chat_postMessage(
                            channel="#rag-pruebas", text=respuesta, concurrent=1)

        finally:
            # Marcar la sesión como finalizada
            session_state[channel_id] = False

    except SlackApiError as e:
        print(f"Error al procesar el mensaje: {e}")


def authorize_google_drive_service_account():
    try:
        service_account_secrets = str(pathlib.Path(
            __file__).parent / "sa_secrets.json")
        # service_account_secrets = "../sa_secrets.json"
        if not os.path.isfile(service_account_secrets):
            abort(500, 'Error interno del servidor. No existe el archivo')
        credentials = sa_credentials.from_service_account_file(
            filename=service_account_secrets,
            scopes=SCOPES
        )
        return credentials, 200
    except Exception as e:
        current_app.logger.error(f'Error durante la autorización: {e}')
        return 'Error al autorizar', 500


@app.route("/load", methods=['POST', 'GET'])
def load_docs_from_drive():
    try:
        data = request.json
        google_drive_folder_path = data.get(
            'folder_path')
        if not google_drive_folder_path:
            return func.HttpResponse(
                '{"msg": "A folder path must be provided in order to load google drive documents"}',
                status_code=500,
                mimetype='application/json'
            )

        creds = authorize_google_drive_service_account()

        service = build('drive', 'v3', credentials=creds)

        folder_id = get_folder_id_from_url(google_drive_folder_path)
        if not os.path.isfile("pickle.pkl"):
            with open('pickle.pkl', 'wb') as pk:
                documents = get_documents_from_folder(service, folder_id)
                pickle.dump(documents, pk)
        else:
            with open('pickle.pkl', 'rb') as pk:
                documents = pickle.load(pk)

        data_elk = []
        metadata_keys = ["mimeType", "name", "last_modifying_user", "size", "description", "title",
                         "_extract_binary_content", "_reduce_whitespace", "_run_ml_inference"]
        for i in documents:
            data_elk.append(
                Document(
                    page_content=i['body'],
                    metadata={k: i.get(k) for k in metadata_keys}
                )
                # i['content']
            )

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512, chunk_overlap=128, model_name="gpt-3.5-turbo"
        )
        docs = text_splitter.transform_documents(data_elk)

        embeddings = ElasticsearchEmbeddings.from_es_connection(
            "sentence-transformers__all-minilm-l6-v2",
            elasticsearch_client,
        )
        batch_size = 20
        print("total: ", len(docs))
        for i in tqdm(range(0, len(docs), batch_size)):
            batch_docs = docs[i:i+batch_size]
            # print(batch_docs)  # Print the batch of documents if needed
            ElasticsearchStore.from_documents(
                batch_docs,
                embeddings,
                es_connection=elasticsearch_client,
                index_name=INDEX,
                bulk_kwargs={
                    "request_timeout": 360,
                    "max_retries": 5,
                    "chunk_size": 500

                },
            )

        return func.HttpResponse(
            "Loads docs Succesfully",
            status_code=200
        )
    except elastic_transport.ConnectionTimeout as e:
        traceback.print_exc()
        # load_docs_from_drive()
    except:
        traceback.print_exc()


def get_elasticsearch_chat_message_history(index, session_id):
    return ElasticsearchChatMessageHistory(
        es_connection=elasticsearch_client, index=index, session_id=session_id
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    request_json = request.get_json()

    # Validar si la pregunta está presente en el JSON de la solicitud
    question = request_json.get("question")
    if not question:
        return func.HttpResponse(
            json.dumps({"error": "Missing question from request JSON"}),
            status_code=400,
            mimetype='application/json'
        )

    # Generar un ID de sesión único si no se proporciona uno en los parámetros
    session_id = request.args.get("session_id", str(uuid4()))

    # Procesar la pregunta y obtener la respuesta
    response = ask_question(question, session_id)
    response = "".join(response)
    # Devolver la respuesta en formato JSON con el código de estado HTTP 200
    return response, 200


def get_folder_id_from_url(url: str):
    url_path = urlparse(url).path
    folder_id = url_path.split("/")[-1]
    return folder_id


def get_documents_from_folder(service, folder_id):
    folders_to_process = deque([folder_id])
    documents = []

    while folders_to_process:
        current_folder = folders_to_process.popleft()
        items = list_files_in_folder(service, current_folder)

        for item in tqdm(items):
            mime_type = item.get("mimeType", "")
            name = item.get("name", "")
            lastModifyingUser = item.get("lastModifyingUser", "")
            size = item.get("size", "")
            description = item.get("description", "")

            if mime_type == "application/vnd.google-apps.folder":
                folders_to_process.append(item["id"])
            elif mime_type == "application/vnd.google-apps.shortcut":
                folders_to_process.append(item["shortcutDetails"]["targetId"])
            elif mime_type in ["application/vnd.google-apps.document", "application/pdf"]:

                file_metadata = service.files().get(
                    fileId=item["id"]).execute()
                mime_type = file_metadata.get("mimeType", "")

                if mime_type == "application/vnd.google-apps.document":
                    doc = service.files().export(
                        fileId=item["id"], mimeType="text/plain").execute()
                    content = doc.decode("utf-8")
                elif mime_type == "application/pdf":
                    pdf_file = download_pdf(service, item["id"])
                    content = extract_pdf_text(pdf_file)

                if len(content) > 0:
                    documents.append(
                        {
                            "mimeType": mime_type,
                            "name": name,
                            "body": content,
                            "title": "ingesta",
                            "last_modifying_user": lastModifyingUser,
                            "size": size,
                            "description": description,
                            "_extract_binary_content": True,
                            "_reduce_whitespace": True,
                            "_run_ml_inference": True
                        }
                    )
    return documents


def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query,
                                   fields="nextPageToken, files(id, name, mimeType, webViewLink, lastModifyingUser, description, size, shortcutDetails)").execute()
    # fields="nextPageToken, files(*)").execute()
    items = results.get("files", [])
    return items


def download_pdf(service, file_id):
    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO(request.execute())
    return file


def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
        text = text.replace("●", " ").replace("-", " ").replace("\n", " ")

    return text
