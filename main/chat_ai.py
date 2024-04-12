from elasticsearch_client import (
    elasticsearch_client,
    get_elasticsearch_chat_message_history,
)
import os
import pathlib
import sys
import azure.functions as func
from flask import render_template, stream_with_context, current_app
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import ElasticsearchEmbeddings
from langchain_elasticsearch._utilities import (
    DistanceStrategy
)
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, dir_path)
# import anonymize

langchain.debug = False

INDEX = os.getenv("ES_INDEX", "workplace-app-docs")
API_KEI_OPENAI = os.getenv("OPENAI_API_KEY", "workplace-app-docs")
INDEX_CHAT_HISTORY = os.getenv(
    "ES_INDEX_CHAT_HISTORY", "workplace-app-docs-chat-history"
)
ELSER_MODEL = os.getenv("ELSER_MODEL", ".elser_model_2")
SESSION_ID_TAG = "[SESSION_ID]"
SOURCE_TAG = "[SOURCE]"
DONE_TAG = "[DONE]"

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("API_VERSION")
AZURE_OPENAI_MODEL_VERSION = os.getenv("MODEL_VERSION")

os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_MODEL_VERSION"] = AZURE_OPENAI_MODEL_VERSION
embeddings = ElasticsearchEmbeddings.from_es_connection(
    "sentence-transformers__all-minilm-l6-v2",
    elasticsearch_client,
)
store = ElasticsearchStore(
    embedding=embeddings,
    es_connection=elasticsearch_client,
    index_name=INDEX,
    strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(
        model_id=ELSER_MODEL,
    ),
)
store.as_retriever()


def init_openai_chat(temperature):
    """
    Inicializa un objeto de chat OpenAI para interactuar con el modelo de lenguaje GPT-3.5.

    Args:
        temperature (float): El parámetro de temperatura utilizado para la generación de texto.
                             Un valor más alto produce respuestas más diversas pero menos coherentes,
                             mientras que un valor más bajo produce respuestas más predecibles pero menos variadas.

    Returns:
        AzureChatOpenAI: Un objeto de chat OpenAI configurado con los parámetros especificados.

    """
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=temperature,
        model="gpt-3.5-turbo",
        azure_deployment="gpt-35-turbo"
    )


@stream_with_context
def ask_question(question, session_id):
    """
    Responde a una pregunta dada utilizando un modelo de lenguaje de inteligencia artificial.

    Args:
        question (str): La pregunta que se desea responder.
        session_id (str): El identificador de la sesión de chat.

    Returns:
        str: La respuesta generada por el modelo de lenguaje.

    Raises:
        Algunas excepciones que podrían ocurrir durante la ejecución del código.

    """
    # Enviar información de la sesión al cliente
    # yield f"data: {SESSION_ID_TAG} {session_id}\n\n"
    # Registrar ID de sesión en el registro de la aplicación
    current_app.logger.debug("Chat session ID: %s", session_id)

    # Obtener el historial del chat desde Elasticsearch
    chat_history = get_elasticsearch_chat_message_history(
        INDEX_CHAT_HISTORY, session_id
    )

    # Inicializar el modelo de lenguaje de OpenAI
    llm = init_openai_chat(0.5)
    if len(chat_history.messages) > 0:
        # Crear una pregunta condensada
        condense_question_prompt = render_template(
            "condense_question_prompt.txt",
            # "condense_question_prompt.txt",
            question=question,
            chat_history=chat_history.messages,
        )
        condensed_question = llm.invoke(condense_question_prompt).content
    else:
        condensed_question = question

    current_app.logger.debug("Condensed question: %s", condensed_question)
    current_app.logger.debug("Question: %s", question)

    # Procesar las palabras clave de la pregunta para la búsqueda
    stop_words = set(stopwords.words('spanish'))
    word_tokens = word_tokenize(question)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    question_tokenized = ' '.join(filtered_sentence)

    # Obtener la incrustación de la pregunta para la búsqueda en Elasticsearch
    docs_embed = embeddings.embed_query(question)

    # Realizar la búsqueda en Elasticsearch
    docs = ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True,
                                                      query_model_id="sentence-transformers__all-minilm-l6-v2", rrf={"rank_constant": 10,
                                                                                                                     "window_size": 100}).query(
        # vector_query_field="ml.inference.body.predicted_value",
        vector_query_field="vector",
        query_vector=docs_embed,
        k=10,
        # text_field="body",
        text_field="metadata.name",
        filter=[
            {
                "match": {
                    "text": {
                        "query": question_tokenized
                    }
                }
            }
        ],
        similarity=DistanceStrategy.COSINE,
        query=question_tokenized,
        fetch_k=1000,

    )
    # Realizar la consulta final utilizando los resultados de la búsqueda
    response_retrieval = store.client.search(
        index=INDEX,
        knn=docs['knn'],
        query=docs['query'],
        rank=docs['rank']
        # search_type="dfs_query_then_fetch"
    )

    metadata_keys = ["mimeType", "name", "path", "author", "description", "body",
                     "_extract_binary_content", "_reduce_whitespace", "_run_ml_inference"]

    # Procesar los documentos recuperados y prepararlos para el modelo de lenguaje
    documentos = []

    for i in response_retrieval['hits']['hits']:
        body = i['_source']['text']
        body = body.replace(" . ", "").replace("__", "").replace(
            ".", "").replace("●", " ").replace("-", " ").replace("\n", " ")
        word_tokens = word_tokenize(body, language="spanish")
        filtered_sentence = [
            w for w in word_tokens if not w.lower() in stop_words]
        body = ' '.join(filtered_sentence)

        # from anonymize import anonymize_text
        # # --------------------Anonimización -----------------------------
        # body = anonymize_text(body)
        # # --------------------Anonimización -----------------------------

        documentos.append(
            Document(
                page_content=body,
                # metadata={k: i["_source"].get(k) for k in metadata_keys},
                metadata={k: i["_source"]["metadata"].get(
                    k) for k in metadata_keys},
            )
        )
    # from anonymize import anonymize_text
    # question = anonymize_text(question)
    # Dividir los documentos en trozos más pequeños para procesamiento eficiente
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4090, chunk_overlap=0, model_name="gpt-3.5-turbo"
    )

    docs_splitter = text_splitter.transform_documents(documentos)

    # Construir la plantilla de la pregunta y documentos para el modelo de lenguaje
    #template = str(pathlib.Path(__file__).parent / "templates/template.txt")
    template = "template.txt"

    qa_prompt_v1 = render_template(
        template,
        question=question,
        docs=docs_splitter[:5],
        chat_history=chat_history.messages,
    )

    # Generar la respuesta utilizando el modelo de lenguaje en streaming

    answer = ""
    for chunk in llm.stream(qa_prompt_v1):
        # chunk.content
        answer += chunk.content

    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)

    return answer
