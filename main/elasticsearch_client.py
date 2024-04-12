from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchChatMessageHistory
from dotenv import load_dotenv
import os
load_dotenv()

ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")
ELASTIC_API_ID = os.getenv("ELASTIC_API_ID")
ELASTIC_USER=os.getenv("ELASTIC_USER")
ELASTIC_PASSWORD=os.getenv("ELASTIC_PASSWORD")

if ELASTICSEARCH_URL:
    elasticsearch_client = Elasticsearch(
        hosts=[ELASTICSEARCH_URL],
        basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD)
    )
elif ELASTIC_CLOUD_ID:
    elasticsearch_client = Elasticsearch(
        cloud_id=ELASTIC_CLOUD_ID, api_key=(ELASTIC_API_ID, ELASTIC_API_KEY), timeout=70, 
    )
else:
    raise ValueError(
        "Please provide either ELASTICSEARCH_URL or ELASTIC_CLOUD_ID and ELASTIC_API_KEY"
    )


def get_elasticsearch_chat_message_history(index, session_id):
    return ElasticsearchChatMessageHistory(
        es_connection=elasticsearch_client, index=index, session_id=session_id
    )
