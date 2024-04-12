import os
import re
from operator import itemgetter

import langchain
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from presidio_analyzer import PatternRecognizer, Pattern
load_dotenv(find_dotenv())
import chat_ai

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

nlp_config = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_trf"},
        {"lang_code": "es", "model_name": "es_dep_news_trf"}

    ],
}
anonymizer = PresidioReversibleAnonymizer(
    # analyzed_fields=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD"],
    languages_config=nlp_config,
    add_default_faker_operators=True,
)

document_id_pattern = Pattern(
    name="doc_id_pattern",
    regex=r"\d{1,3}(?:\.\d{3})*",
    score=1,
)
document_id_recognizer = PatternRecognizer(
    supported_entity="DOCUMENT_ID", patterns=[document_id_pattern]
)

anonymizer.add_recognizer(document_id_recognizer)
anonymizer.reset_deanonymizer_mapping()


def anonimizador(text: str, question: str) -> str:
    anonimo = anonymizer.anonymize(text, language="en")
    print(anonymizer.deanonymizer_mapping)
    return anonimo


print()


def print_colored_pii(string):
    colored_string = re.sub(
        r"(<[^>]*>)", lambda m: "\033[31m" + m.group(1) + "\033[0m", string
    )
    print(colored_string)


def anonymize_text(text):

    text_anonimize = anonymizer.anonymize(text.lower())


    return text_anonimize

