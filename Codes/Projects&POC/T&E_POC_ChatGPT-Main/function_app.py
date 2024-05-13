import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
import pandas as pd
import copy
import json
import os
import logging
import uuid
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI ,AzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from backend.auth.auth_utils import get_authenticated_user_details
from backend.history.cosmosdbservice import CosmosConversationClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
# from azure.identity import DefaultAzureCredential,get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from backend.utils import format_as_ndjson, format_stream_response, generateFilterString, parse_multi_columns, format_non_streaming_response
from backend.admin import insertuser,getuser,updateuser
from backend.admin import insertentity,getentity,updateentity
from backend.admin import insertrole,getrole,updaterole
from backend.admin import insertquery,updatequery,vector_search,getquerydetails
from collections import namedtuple
from datetime import datetime
import functools
import json
import six
from functools import lru_cache
#import cachetools
load_dotenv()


Serialized = namedtuple('Serialized', 'json')


# def hashable_cache(cache):
#     def hashable_cache_internal(func):
#         def deserialize(value):
#             if isinstance(value, Serialized):
#                 return json.loads(value.json)
#             else:
#                 return value

#         def func_with_serialized_params(*args, **kwargs):
#             _args = tuple([deserialize(arg) for arg in args])
#             _kwargs = {k: deserialize(v) for k, v in six.viewitems(kwargs)}
#             return func(*_args, **_kwargs)

#         cached_func = cache(func_with_serialized_params)

#         @functools.wraps(func)
#         def hashable_cached_func(*args, **kwargs):
#             _args = tuple([
#                 Serialized(json.dumps(arg, sort_keys=True))
#                 if type(arg) in (list, dict) else arg
#                 for arg in args
#             ])
#             _kwargs = {
#                 k: Serialized(json.dumps(v, sort_keys=True))
#                 if type(v) in (list, dict) else v
#                 for k, v in kwargs.items()
#             }
#             return cached_func(*_args, **_kwargs)
#         hashable_cached_func.cache_info = cached_func.cache_info
#         hashable_cached_func.cache_clear = cached_func.cache_clear
#         return hashable_cached_func

#     return hashable_cache_internal

# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

USER_AGENT = "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"

# On Your Data Settings
DATASOURCE_TYPE = os.environ.get("DATASOURCE_TYPE", "AzureCognitiveSearch")
SEARCH_TOP_K = os.environ.get("SEARCH_TOP_K", 5)
SEARCH_STRICTNESS = os.environ.get("SEARCH_STRICTNESS", 3)
SEARCH_ENABLE_IN_DOMAIN = os.environ.get("SEARCH_ENABLE_IN_DOMAIN", "true")

# ACS Integration Settings
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", None)
AZURE_SEARCH_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_USE_SEMANTIC_SEARCH", "false")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG", "default")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K", SEARCH_TOP_K)
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS")
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN")
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN")
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN")
AZURE_SEARCH_VECTOR_COLUMNS = os.environ.get("AZURE_SEARCH_VECTOR_COLUMNS")
AZURE_SEARCH_QUERY_TYPE = os.environ.get("AZURE_SEARCH_QUERY_TYPE")
AZURE_SEARCH_PERMITTED_GROUPS_COLUMN = os.environ.get("AZURE_SEARCH_PERMITTED_GROUPS_COLUMN")
AZURE_SEARCH_STRICTNESS = os.environ.get("AZURE_SEARCH_STRICTNESS", SEARCH_STRICTNESS)

# AOAI Integration Settings
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE", 0)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P", 1.0)
AZURE_OPENAI_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS", 1000)
AZURE_OPENAI_STOP_SEQUENCE = os.environ.get("AZURE_OPENAI_STOP_SEQUENCE")
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE", "You are an AI assistant that helps people find information.")
AZURE_OPENAI_PREVIEW_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2023-12-01-preview")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM", "true")
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo-16k") # Name of the model, e.g. 'gpt-35-turbo-16k' or 'gpt-4'
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY")
AZURE_OPENAI_EMBEDDING_NAME = os.environ.get("AZURE_OPENAI_EMBEDDING_NAME", "")

# CosmosDB Mongo vcore vector db Settings
AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING")  #This has to be secure string
AZURE_COSMOSDB_MONGO_VCORE_DATABASE = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_DATABASE")
AZURE_COSMOSDB_MONGO_VCORE_CONTAINER = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONTAINER")
AZURE_COSMOSDB_MONGO_VCORE_INDEX = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_INDEX")
AZURE_COSMOSDB_MONGO_VCORE_TOP_K = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_TOP_K", AZURE_SEARCH_TOP_K)
AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS", AZURE_SEARCH_STRICTNESS)  
AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN", AZURE_SEARCH_ENABLE_IN_DOMAIN)
AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS", "")
AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS")

SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# Chat History CosmosDB Integration Settings
AZURE_COSMOSDB_DATABASE = os.environ.get("AZURE_COSMOSDB_DATABASE")
AZURE_COSMOSDB_ACCOUNT = os.environ.get("AZURE_COSMOSDB_ACCOUNT")
AZURE_COSMOSDB_CONVERSATIONS_CONTAINER = os.environ.get("AZURE_COSMOSDB_CONVERSATIONS_CONTAINER")
AZURE_COSMOSDB_ACCOUNT_KEY = os.environ.get("AZURE_COSMOSDB_ACCOUNT_KEY")
AZURE_COSMOSDB_ENABLE_FEEDBACK = os.environ.get("AZURE_COSMOSDB_ENABLE_FEEDBACK", "false").lower() == "true"

# Elasticsearch Integration Settings
ELASTICSEARCH_ENDPOINT = os.environ.get("ELASTICSEARCH_ENDPOINT")
ELASTICSEARCH_ENCODED_API_KEY = os.environ.get("ELASTICSEARCH_ENCODED_API_KEY")
ELASTICSEARCH_INDEX = os.environ.get("ELASTICSEARCH_INDEX")
ELASTICSEARCH_QUERY_TYPE = os.environ.get("ELASTICSEARCH_QUERY_TYPE", "simple")
ELASTICSEARCH_TOP_K = os.environ.get("ELASTICSEARCH_TOP_K", SEARCH_TOP_K)
ELASTICSEARCH_ENABLE_IN_DOMAIN = os.environ.get("ELASTICSEARCH_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
ELASTICSEARCH_CONTENT_COLUMNS = os.environ.get("ELASTICSEARCH_CONTENT_COLUMNS")
ELASTICSEARCH_FILENAME_COLUMN = os.environ.get("ELASTICSEARCH_FILENAME_COLUMN")
ELASTICSEARCH_TITLE_COLUMN = os.environ.get("ELASTICSEARCH_TITLE_COLUMN")
ELASTICSEARCH_URL_COLUMN = os.environ.get("ELASTICSEARCH_URL_COLUMN")
ELASTICSEARCH_VECTOR_COLUMNS = os.environ.get("ELASTICSEARCH_VECTOR_COLUMNS")
ELASTICSEARCH_STRICTNESS = os.environ.get("ELASTICSEARCH_STRICTNESS", SEARCH_STRICTNESS)
ELASTICSEARCH_EMBEDDING_MODEL_ID = os.environ.get("ELASTICSEARCH_EMBEDDING_MODEL_ID")

# Pinecone Integration Settings
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_TOP_K = os.environ.get("PINECONE_TOP_K", SEARCH_TOP_K)
PINECONE_STRICTNESS = os.environ.get("PINECONE_STRICTNESS", SEARCH_STRICTNESS)  
PINECONE_ENABLE_IN_DOMAIN = os.environ.get("PINECONE_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
PINECONE_CONTENT_COLUMNS = os.environ.get("PINECONE_CONTENT_COLUMNS", "")
PINECONE_FILENAME_COLUMN = os.environ.get("PINECONE_FILENAME_COLUMN")
PINECONE_TITLE_COLUMN = os.environ.get("PINECONE_TITLE_COLUMN")
PINECONE_URL_COLUMN = os.environ.get("PINECONE_URL_COLUMN")
PINECONE_VECTOR_COLUMNS = os.environ.get("PINECONE_VECTOR_COLUMNS")

# Azure AI MLIndex Integration Settings - for use with MLIndex data assets created in Azure AI Studio
AZURE_MLINDEX_NAME = os.environ.get("AZURE_MLINDEX_NAME")
AZURE_MLINDEX_VERSION = os.environ.get("AZURE_MLINDEX_VERSION")
AZURE_ML_PROJECT_RESOURCE_ID = os.environ.get("AZURE_ML_PROJECT_RESOURCE_ID") # /subscriptions/{sub ID}/resourceGroups/{rg name}/providers/Microsoft.MachineLearningServices/workspaces/{AML project name}
AZURE_MLINDEX_TOP_K = os.environ.get("AZURE_MLINDEX_TOP_K", SEARCH_TOP_K)
AZURE_MLINDEX_STRICTNESS = os.environ.get("AZURE_MLINDEX_STRICTNESS", SEARCH_STRICTNESS)  
AZURE_MLINDEX_ENABLE_IN_DOMAIN = os.environ.get("AZURE_MLINDEX_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
AZURE_MLINDEX_CONTENT_COLUMNS = os.environ.get("AZURE_MLINDEX_CONTENT_COLUMNS", "")
AZURE_MLINDEX_FILENAME_COLUMN = os.environ.get("AZURE_MLINDEX_FILENAME_COLUMN")
AZURE_MLINDEX_TITLE_COLUMN = os.environ.get("AZURE_MLINDEX_TITLE_COLUMN")
AZURE_MLINDEX_URL_COLUMN = os.environ.get("AZURE_MLINDEX_URL_COLUMN")
AZURE_MLINDEX_VECTOR_COLUMNS = os.environ.get("AZURE_MLINDEX_VECTOR_COLUMNS")
AZURE_MLINDEX_QUERY_TYPE = os.environ.get("AZURE_MLINDEX_QUERY_TYPE")


# Frontend Settings via Environment Variables
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() == "true"
CHAT_HISTORY_ENABLED = AZURE_COSMOSDB_ACCOUNT and AZURE_COSMOSDB_DATABASE and AZURE_COSMOSDB_CONVERSATIONS_CONTAINER

endpoint = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.environ.get("AZURE_QUERY_INDEX")
azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
embedding_model_name = AZURE_OPENAI_EMBEDDING_NAME
credential = AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]) if len(os.environ["AZURE_SEARCH_KEY"]) > 0 else DefaultAzureCredential()
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

openai_credential = DefaultAzureCredential()
# token_provider = get_bearer_token_provider(openai_credential, "https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    azure_deployment=AZURE_OPENAI_EMBEDDING_NAME,
    api_version=azure_openai_api_version,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY
    # ,azure_ad_token_provider=token_provider if not AZURE_OPENAI_KEY else None
)

def should_use_data():
    global DATASOURCE_TYPE
    if AZURE_SEARCH_SERVICE and AZURE_SEARCH_INDEX:
        DATASOURCE_TYPE = "AzureCognitiveSearch"
        logging.debug("Using Azure Cognitive Search")
        return True
    
    if AZURE_COSMOSDB_MONGO_VCORE_DATABASE and AZURE_COSMOSDB_MONGO_VCORE_CONTAINER and AZURE_COSMOSDB_MONGO_VCORE_INDEX and AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING:
        DATASOURCE_TYPE = "AzureCosmosDB"
        logging.debug("Using Azure CosmosDB Mongo vcore")
        return True
    
    if ELASTICSEARCH_ENDPOINT and ELASTICSEARCH_ENCODED_API_KEY and ELASTICSEARCH_INDEX:
        DATASOURCE_TYPE = "Elasticsearch"
        logging.debug("Using Elasticsearch")
        return True
    
    if PINECONE_ENVIRONMENT and PINECONE_API_KEY and PINECONE_INDEX_NAME:
        DATASOURCE_TYPE = "Pinecone"
        logging.debug("Using Pinecone")
        return True
    
    if AZURE_MLINDEX_NAME and AZURE_MLINDEX_VERSION and AZURE_ML_PROJECT_RESOURCE_ID:
        DATASOURCE_TYPE = "AzureMLIndex"
        logging.debug("Using Azure ML Index")
        return True

    return False

SHOULD_USE_DATA = should_use_data()

# Initialize Azure OpenAI Client
def init_openai_client(use_data=SHOULD_USE_DATA):
    azure_openai_client = None
    try:
        # Endpoint
        if not AZURE_OPENAI_ENDPOINT and not AZURE_OPENAI_RESOURCE:
            raise Exception("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE is required")
        
        endpoint = AZURE_OPENAI_ENDPOINT if AZURE_OPENAI_ENDPOINT else f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
        
        # Authentication
        aoai_api_key = AZURE_OPENAI_KEY
        ad_token_provider = None
        if not aoai_api_key:
            logging.debug("No AZURE_OPENAI_KEY found, using Azure AD auth")
            ad_token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

        # Deployment
        deployment = AZURE_OPENAI_MODEL
        if not deployment:
            raise Exception("AZURE_OPENAI_MODEL is required")

        # Default Headers
        default_headers = {
            'x-ms-useragent': USER_AGENT
        }

        if use_data:
            #https://travel-openai.openai.azure.com/openai/deployments/travel-model/chat/completions?api-version=2024-02-15-preview  
            base_url = f"{str(endpoint).rstrip('/')}/openai/deployments/{deployment}/extensions"
            #azure_openai_client = AsyncAzureOpenAI(
            azure_openai_client = AzureOpenAI(              
                base_url=str(base_url),                
                #azure_deployment=deployment,
                api_version= "2023-08-01-preview", 
                ##AZURE_OPENAI_PREVIEW_API_VERSION,
                api_key=aoai_api_key,
                #azure_ad_token_provider=ad_token_provider,
                #default_headers=default_headers,                
            )
        else:
            azure_openai_client = AsyncAzureOpenAI(
                api_version=AZURE_OPENAI_PREVIEW_API_VERSION,
                api_key=aoai_api_key,
                azure_ad_token_provider=ad_token_provider,
                default_headers=default_headers,
                azure_endpoint=endpoint
            )
            
        return azure_openai_client
    except Exception as e:
        logging.exception("Exception in Azure OpenAI initialization", e)
        azure_openai_client = None
        raise e


def init_cosmosdb_client():
    cosmos_conversation_client = None
    if CHAT_HISTORY_ENABLED:
        try:
            cosmos_endpoint = f'https://{AZURE_COSMOSDB_ACCOUNT}.documents.azure.com:443/'

            if not AZURE_COSMOSDB_ACCOUNT_KEY:
                credential = DefaultAzureCredential()
            else:
                credential = AZURE_COSMOSDB_ACCOUNT_KEY

            cosmos_conversation_client = CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint, 
                credential=credential, 
                database_name=AZURE_COSMOSDB_DATABASE,
                container_name=AZURE_COSMOSDB_CONVERSATIONS_CONTAINER,
                enable_message_feedback=AZURE_COSMOSDB_ENABLE_FEEDBACK
            )
        except Exception as e:
            logging.exception("Exception in CosmosDB initialization", e)
            cosmos_conversation_client = None
            raise e
    else:
        logging.debug("CosmosDB not configured")
        
    return cosmos_conversation_client


def get_configured_data_source():
    data_source = {}
    query_type = "simple"
    if DATASOURCE_TYPE == "AzureCognitiveSearch":
        # Set query type
        if AZURE_SEARCH_QUERY_TYPE:
            query_type = AZURE_SEARCH_QUERY_TYPE
        elif AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" and AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG:
            query_type = "semantic"

        # Set filter
        filter = None
        userToken = None
        
        if AZURE_SEARCH_PERMITTED_GROUPS_COLUMN:
            userToken = req.headers.get('X-MS-TOKEN-AAD-ACCESS-TOKEN', "")
            logging.debug(f"USER TOKEN is {'present' if userToken else 'not present'}")

            filter = generateFilterString(userToken)
            logging.debug(f"FILTER: {filter}")
        
        # Set authentication
        authentication = {}
        if AZURE_SEARCH_KEY:
            authentication = {
                "type": "APIKey",
                "key": AZURE_SEARCH_KEY,
                "apiKey": AZURE_SEARCH_KEY
            }
        else:
            # If key is not provided, assume AOAI resource identity has been granted access to the search service
            authentication = {
                "type": "SystemAssignedManagedIdentity"
            }

        data_source = {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
                    #"authentication": authentication,
                    "key": AZURE_SEARCH_KEY,
                    "indexName": AZURE_SEARCH_INDEX,    
                    "fieldsMapping": {
                        "contentFields": parse_multi_columns(AZURE_SEARCH_CONTENT_COLUMNS) if AZURE_SEARCH_CONTENT_COLUMNS else [],
                        "titleField": AZURE_SEARCH_TITLE_COLUMN if AZURE_SEARCH_TITLE_COLUMN else None,
                        "urlField": AZURE_SEARCH_URL_COLUMN if AZURE_SEARCH_URL_COLUMN else None,
                        "filepathField": AZURE_SEARCH_FILENAME_COLUMN if AZURE_SEARCH_FILENAME_COLUMN else None,
                        "vectorFields": parse_multi_columns(AZURE_SEARCH_VECTOR_COLUMNS) if AZURE_SEARCH_VECTOR_COLUMNS else []
                    },
                     "inScope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                     "topNDocuments": int(AZURE_SEARCH_TOP_K) if AZURE_SEARCH_TOP_K else int(SEARCH_TOP_K),
                     "queryType": query_type,
                     "semanticConfiguration": AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG if AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG else "",
                     "embeddingEndpoint": 'https://travel-openai.openai.azure.com/openai/deployments/travel-ada/embeddings?api-version=2023-08-01-preview',                      
                     "embeddingKey": '01995a8ce3a34cfdb92c7033af10b24d',
                     "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE,
                     #"filter": filter,
                     "strictness": int(AZURE_SEARCH_STRICTNESS) if AZURE_SEARCH_STRICTNESS else int(SEARCH_STRICTNESS)
                }
            }
    elif DATASOURCE_TYPE == "AzureCosmosDB":
        query_type = "vector"

        data_source = {
                "type": "AzureCosmosDB",
                "parameters": {
                    "authentication": {
                        "type": "ConnectionString",
                        "connectionString": AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING
                    },
                    "indexName": AZURE_COSMOSDB_MONGO_VCORE_INDEX,
                    "databaseName": AZURE_COSMOSDB_MONGO_VCORE_DATABASE,
                    "containerName": AZURE_COSMOSDB_MONGO_VCORE_CONTAINER,                    
                    "fieldsMapping": {
                        "contentFields": parse_multi_columns(AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS) if AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS else [],
                        "titleField": AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN else None,
                        "urlField": AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN else None,
                        "filepathField": AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN else None,
                        "vectorFields": parse_multi_columns(AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS) if AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS else []
                    },
                    "inScope": True if AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": int(AZURE_COSMOSDB_MONGO_VCORE_TOP_K) if AZURE_COSMOSDB_MONGO_VCORE_TOP_K else int(SEARCH_TOP_K),
                    "strictness": int(AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS) if AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS else int(SEARCH_STRICTNESS),
                    "queryType": query_type,
                    "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE
                }
            }
    elif DATASOURCE_TYPE == "Elasticsearch":
        if ELASTICSEARCH_QUERY_TYPE:
            query_type = ELASTICSEARCH_QUERY_TYPE

        data_source = {
            "type": "Elasticsearch",
            "parameters": {
                "endpoint": ELASTICSEARCH_ENDPOINT,
                "authentication": {
                    "type": "EncodedAPIKey",
                    "encodedApiKey": ELASTICSEARCH_ENCODED_API_KEY
                },
                "indexName": ELASTICSEARCH_INDEX,
                "fieldsMapping": {
                    "contentFields": parse_multi_columns(ELASTICSEARCH_CONTENT_COLUMNS) if ELASTICSEARCH_CONTENT_COLUMNS else [],
                    "titleField": ELASTICSEARCH_TITLE_COLUMN if ELASTICSEARCH_TITLE_COLUMN else None,
                    "urlField": ELASTICSEARCH_URL_COLUMN if ELASTICSEARCH_URL_COLUMN else None,
                    "filepathField": ELASTICSEARCH_FILENAME_COLUMN if ELASTICSEARCH_FILENAME_COLUMN else None,
                    "vectorFields": parse_multi_columns(ELASTICSEARCH_VECTOR_COLUMNS) if ELASTICSEARCH_VECTOR_COLUMNS else []
                },
                "inScope": True if ELASTICSEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                "topNDocuments": int(ELASTICSEARCH_TOP_K) if ELASTICSEARCH_TOP_K else int(SEARCH_TOP_K),
                "queryType": query_type,
                "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE,
                "strictness": int(ELASTICSEARCH_STRICTNESS) if ELASTICSEARCH_STRICTNESS else int(SEARCH_STRICTNESS)
            }
        }
    elif DATASOURCE_TYPE == "AzureMLIndex":
        if AZURE_MLINDEX_QUERY_TYPE:
            query_type = AZURE_MLINDEX_QUERY_TYPE

        data_source = {
            "type": "AzureMLIndex",
            "parameters": {
                "name": AZURE_MLINDEX_NAME,
                "version": AZURE_MLINDEX_VERSION,
                "projectResourceId": AZURE_ML_PROJECT_RESOURCE_ID,
                "fieldsMapping": {
                    "contentFields": parse_multi_columns(AZURE_MLINDEX_CONTENT_COLUMNS) if AZURE_MLINDEX_CONTENT_COLUMNS else [],
                    "titleField": AZURE_MLINDEX_TITLE_COLUMN if AZURE_MLINDEX_TITLE_COLUMN else None,
                    "urlField": AZURE_MLINDEX_URL_COLUMN if AZURE_MLINDEX_URL_COLUMN else None,
                    "filepathField": AZURE_MLINDEX_FILENAME_COLUMN if AZURE_MLINDEX_FILENAME_COLUMN else None,
                    "vectorFields": parse_multi_columns(AZURE_MLINDEX_VECTOR_COLUMNS) if AZURE_MLINDEX_VECTOR_COLUMNS else []
                },
                "inScope": True if AZURE_MLINDEX_ENABLE_IN_DOMAIN.lower() == "true" else False,
                "topNDocuments": int(AZURE_MLINDEX_TOP_K) if AZURE_MLINDEX_TOP_K else int(SEARCH_TOP_K),
                "queryType": query_type,
                "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE,
                "strictness": int(AZURE_MLINDEX_STRICTNESS) if AZURE_MLINDEX_STRICTNESS else int(SEARCH_STRICTNESS)
            }
        }
    elif DATASOURCE_TYPE == "Pinecone":
        query_type = "vector"

        data_source = {
            "type": "Pinecone",
            "parameters": {
                "environment": PINECONE_ENVIRONMENT,
                "authentication": {
                    "type": "APIKey",
                    "key": PINECONE_API_KEY
                },
                "indexName": PINECONE_INDEX_NAME,
                "fieldsMapping": {
                    "contentFields": parse_multi_columns(PINECONE_CONTENT_COLUMNS) if PINECONE_CONTENT_COLUMNS else [],
                    "titleField": PINECONE_TITLE_COLUMN if PINECONE_TITLE_COLUMN else None,
                    "urlField": PINECONE_URL_COLUMN if PINECONE_URL_COLUMN else None,
                    "filepathField": PINECONE_FILENAME_COLUMN if PINECONE_FILENAME_COLUMN else None,
                    "vectorFields": parse_multi_columns(PINECONE_VECTOR_COLUMNS) if PINECONE_VECTOR_COLUMNS else []
                },
                "inScope": True, #if PINECONE_ENABLE_IN_DOMAIN.lower() == "true" else False,
                "topNDocuments": int(PINECONE_TOP_K) if PINECONE_TOP_K else int(SEARCH_TOP_K),
                "strictness": int(PINECONE_STRICTNESS) if PINECONE_STRICTNESS else int(SEARCH_STRICTNESS),
                "queryType": query_type,
                "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE,
            }
        }
    else:
        raise Exception(f"DATASOURCE_TYPE is not configured or unknown: {DATASOURCE_TYPE}")

    if "vector" in query_type.lower() and DATASOURCE_TYPE != "AzureMLIndex":
        embeddingDependency = {}
        if AZURE_OPENAI_EMBEDDING_NAME:
            embeddingDependency = {
                "type": "DeploymentName",
                "deploymentName": AZURE_OPENAI_EMBEDDING_NAME
            }
        elif AZURE_OPENAI_EMBEDDING_ENDPOINT and AZURE_OPENAI_EMBEDDING_KEY:
            embeddingDependency = {
                "type": "Endpoint",
                "endpoint": AZURE_OPENAI_EMBEDDING_ENDPOINT,
                "authentication": {
                    "type": "APIKey",
                    "key": AZURE_OPENAI_EMBEDDING_KEY
                }
            }
        elif DATASOURCE_TYPE == "Elasticsearch" and ELASTICSEARCH_EMBEDDING_MODEL_ID:
            embeddingDependency = {
                "type": "ModelId",
                "modelId": ELASTICSEARCH_EMBEDDING_MODEL_ID
            }
        else:
            raise Exception(f"Vector query type ({query_type}) is selected for data source type {DATASOURCE_TYPE} but no embedding dependency is configured")
        data_source["parameters"]["embeddingDependency"] = embeddingDependency

    return data_source

def generate_embeddings(query):
    '''
    Generate embeddings from string of text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    return client.embeddings.create(input=query, model=embedding_model_name).data[0].embedding

def getsearchresult(query):    
    try:    
        embedding = generate_embeddings(query)
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=1, fields="titleVector")
    
        results = search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["Query", "content","Citation"],
        ) 
        # print(results)

        for result in results: 
            # print(result)
            if result and  result['@search.score'] >=0.98:
                # print(f"Query: {result['Query']}")  
                # print(f"Score: {result['@search.score']}")  
                # print(f"Content: {result['content']}")  
                # print(f"Citation: {result['Citation']}\n")
                return result['Citation']
            else: 
                return
    except Exception as e:
        logging.exception("Exception while fetching search results")
        raise e


def cacheresult(id, query,content,result):
    try:
        input_data = [{"id":id,"Query": query,"content": content,"Citation": json.dumps(result)}]

        titles = [item['Query'] for item in input_data]
        content = [item['content'] for item in input_data]
        title_response = client.embeddings.create(input=titles, model=embedding_model_name)
        title_embeddings = [item.embedding for item in title_response.data]
        content_response = client.embeddings.create(input=content, model=embedding_model_name)
        content_embeddings = [item.embedding for item in content_response.data]

        # # Generate embeddings for title and content fields
        for i, item in enumerate(input_data):
            title = item['Query']
            content = item['content']
            item['titleVector'] = title_embeddings[i]
            item['contentVector'] = content_embeddings[i]
        
        search_client.upload_documents(input_data)
        # return len(input_data)
    except Exception as e:
        logging.exception("Exception while caching the query")
        raise e

                
def prepare_model_args(request_body):
    request_messages = request_body.get("messages", [])
    messages = []
    if not SHOULD_USE_DATA:
        messages = [
            {
                "role": "system",
                "content": AZURE_OPENAI_SYSTEM_MESSAGE
            }
        ]

    for message in request_messages:
        if message:
            messages.append({
                "role": message["role"] ,
                "content": message["content"]
            })

    model_args = {
        "messages": messages,
        "temperature": float(AZURE_OPENAI_TEMPERATURE),
        "max_tokens": int(AZURE_OPENAI_MAX_TOKENS),
        "top_p": float(AZURE_OPENAI_TOP_P),
        "stop": parse_multi_columns(AZURE_OPENAI_STOP_SEQUENCE) if AZURE_OPENAI_STOP_SEQUENCE else None,
        "stream": SHOULD_STREAM,
        "model": AZURE_OPENAI_MODEL,
        "seed": 42,
    }

    if SHOULD_USE_DATA:
        model_args["extra_body"] = {
            "dataSources": [get_configured_data_source()]
        }
    return model_args

def send_chat_request(request):    
    model_args = prepare_model_args(request)
    #print(model_args)
    try:
        azure_openai_client = init_openai_client()
        response = azure_openai_client.chat.completions.create(**model_args)  
    
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        raise e

    return response

def savequery(query,empid,category,result):
    try:
        query_embedding = generate_embeddings(query)   

        # savequery  
        content =result["choices"][0]["messages"][1]["content"]                 
        referencecontent = json.loads(result["choices"][0]["messages"][0]["content"])["citations"]
        response_obj = {
                "Empid": empid,            
                "Category" : category,
                "Query": query,
                "QueryVector" : query_embedding,
                "Content": content,
                "Citation":referencecontent,
                "ThumbsUp/Down": 1,
                "Datetime":datetime.now()
                }  
        insertquery(response_obj)
    except Exception as e:
        logging.exception("Exception while saving query")
        raise e


def complete_chat_request(request_body):
    # request_messages = request_body.get("messages", [])
    # query = ''

    # for message in request_messages:
    #     if message:
    #         query = message["content"]
    #         empid = message["Empid"]
    #         category = message["Category"]

    query = request_body['messages'][0]['content']
    category = request_body['messages'][0]['Category']
    empid = request_body['messages'][0]['Empid']
    
    response =getsearchresult(query)  

    if response is None:  
        response = send_chat_request(request_body)
        history_metadata = request_body.get("history_metadata", {})

        result = format_non_streaming_response(response, history_metadata)
        id =result['id']
        content =result["choices"][0]["messages"][1]["content"]
       
        cacheresult(id,query,content,result)
          
        savequery(query,empid,category,result)       
            
        return json.dumps(result)
    else:
        result = json.loads(response)
        # print(result)
        # print(response)
        savequery(query,empid,category,result) 
        return response
        

def stream_chat_request(request_body):
    response = send_chat_request(request_body)
    history_metadata = request_body.get("history_metadata", {})

    def generate():
        for completionChunk in response:
            yield format_stream_response(completionChunk, history_metadata)

    return generate()

from cachetools.func import lru_cache

#@hashable_cache(lru_cache())
def conversation_internal(request_body):
    try:
       
        result =  complete_chat_request(request_body)
        return result
    
    except Exception as ex:
        logging.exception(ex)
        if ex.status_code:
           return func.HttpResponse(json.dumps({"error": str(ex)}), ex.status_code)
        else:
           return func.HttpResponse(json.dumps({"error": str(ex)}), 500)

##Query Details##
@app.route("querydetails/insert", methods=["POST"])
def add_query(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json() 
        #print(request_json)
        response = insertquery(request_json)       
        return func.HttpResponse(response)
    except Exception as e:
        logging.exception("Exception in querydetails/isert")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)
    
@app.route("querydetails/update", methods=["POST"])
def update_query(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json() 
        #print(request_json)
        response = updatequery(request_json)
        #response = 'User inserted successfully'
        return func.HttpResponse(response)
    except Exception as e:
        logging.exception("Exception in querydetails/update")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)
    
@app.route("querydetails/get", methods=["POST","GET"])
def get_queries(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json() 
        # request_messages = request_json.get("messages", [])
        # query = ''

        # for message in request_messages:
        #     if message:
        #         query = message["content"]                
        #         category = message["Category"]
        query = request_json['messages'][0]['content']
        category = request_json['messages'][0]['Category']

        query_embedding = generate_embeddings(query)
        results = vector_search(query_embedding)
        response =[]
        for result in results: 
            if result['document']['Category'] == category and result['document']['ThumbsUp/Down'] ==1:
                response.append(result['document']['Query'])
                # print(f"Similarity Score: {result['similarityScore']}")  
                # print(f"Title: {result['document']['title']}")  
                # print(f"Content: {result['document']['content']}")  
                # print(f"Category: {result['document']['category']}\n") 

        unique_queries = []
 
        # traverse for all elements
        for x in response:
        # check if exists in unique_list or not
            if x not in unique_queries:
                unique_queries.append(x)       
        return func.HttpResponse(json.dumps(unique_queries),mimetype="application/json")

    except Exception as e:
        logging.exception("Exception in querydetails/get")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

@app.route("dashboarddata/get", methods=["POST","GET"])
def get_dashboarddata(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json() 
        # request_messages = request_json.get("messages", [])
        # query = ''

        # for message in request_messages:
        #     if message:
        #         startdate = message["startdate"]                
        #         enddate = message["enddate"]

        startdate = request_json['messages'][0]['startdate']
        enddate = request_json['messages'][0]['enddate']

        results = getquerydetails(startdate,enddate)
        print(results)
        df = pd.DataFrame(results)
        print(df)
        df['Date'] = pd.to_datetime(df['Datetime']).dt.date
        df.drop('Datetime',axis=1,inplace=True)        
        print(df)
        data = df.groupby(['Date','Empid','Category']).count()
        print(data)  
        data['ThumbsUp'] = df.groupby(['Date','Empid','Category'])['ThumbsUp/Down'].sum()
        data['Thumbsdown'] = data['ThumbsUp/Down'] - data['ThumbsUp']
        print(data) 
        response = data.to_dict(orient='list')
        json_data = data.to_json(orient='records')
        print(json_data)
        return func.HttpResponse(json.dumps(json_data),mimetype="application/json")
        
    except Exception as e:
        logging.exception("Exception in querydetails/get")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)   
##User Details##
@app.route("userdetails/insert", methods=["POST"])
def add_user(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json() 
        # print(request_json)
        response = insertuser(request_json)       
        return func.HttpResponse(response)
    except Exception as e:
        logging.exception("Exception in userdetails/insert")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

@app.route("userdetails/update", methods=["POST"])
def update_user(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json() 
        # print(request_json)
        response = updateuser(request_json)
        #response = 'User inserted successfully'
        return func.HttpResponse(response)
    except Exception as e:
        logging.exception("Exception in userdetails/update")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

@app.route("userdetails/get", methods=["GET","POST"])
def get_user(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json() 
        #print(request_json)
        response = getuser(request_json)
        #response = 'User fetched successfully'
        return func.HttpResponse(json.dumps(response),mimetype="application/json")
    except Exception as e:
        logging.exception("Exception in /userdetails/get")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

##Entity Details##    
@app.route("entitydetails/insert", methods=["POST"])
def add_entity(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json() 
        # print(request_json)
        response = insertentity(request_json)
        #response = 'User inserted successfully'
        return func.HttpResponse(response)
    except Exception as e:
        logging.exception("Exception in /entitydetails/insert")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

@app.route("entitydetails/update", methods=["POST"])
def update_entity(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json()         
        response = updateentity(request_json)
        #response = 'User inserted successfully'
        return func.HttpResponse(response)
    except Exception as e:
        logging.exception("Exception in /entitydetails/update")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

@app.route("entitydetails/get", methods=["GET","POST"])
def get_entity(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json() 
        logging.info("getting entity details")       
        response = getentity(request_json)
        #response = 'User fetched successfully'
        return func.HttpResponse(json.dumps(response),mimetype="application/json")
    except Exception as e:
        logging.exception("Exception in /entitydetails/get")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)
    
##Role Details##    
@app.route("roledetails/insert", methods=["POST"])
def add_role(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json()         
        response = insertrole(request_json)
        #response = 'User inserted successfully'
        return func.HttpResponse(response)
    except Exception as e:
        logging.exception("Exception in /roledetails/insert")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

@app.route("roledetails/update", methods=["POST"])
def update_role(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json()         
        response = updaterole(request_json)        
        return func.HttpResponse(response)
    except Exception as e:
        logging.exception("Exception in /roledetails/update")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

@app.route("roledetails/get", methods=["GET","POST"])
def get_role(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json()         
        response = getrole(request_json)        
        return func.HttpResponse(json.dumps(response),mimetype="application/json")
    except Exception as e:
        logging.exception("Exception in /roledetails/get")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)
    
#cache = cachetools.LRUCache(maxsize= 128)

@app.route("conversation", methods=["POST"])

def conversation(req: func.HttpRequest) -> func.HttpResponse:
    try:
        request_json = req.get_json()
        
        result = conversation_internal(request_json)
        #print(conversation_internal.cache_info())
            #cache[request_json] = result
        return func.HttpResponse(result)
    except Exception as e:
        logging.exception("Exception in conversation api")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)



## Conversation History API ## 
@app.route("history/generate", methods=["POST"])
def add_conversation(req: func.HttpRequest) -> func.HttpResponse:
    authenticated_user = get_authenticated_user_details(request_headers=req.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    request_json =  req.get_json()
    conversation_id = request_json.get('conversation_id', None)

    try:
        # make sure cosmos is configured
        cosmos_conversation_client = init_cosmosdb_client()
        if not cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        history_metadata = {}
        if not conversation_id:
            title =  generate_title(request_json["messages"])
            conversation_dict = cosmos_conversation_client.create_conversation(user_id=user_id, title=title)
            conversation_id = conversation_dict['id']
            history_metadata['title'] = title
            history_metadata['date'] = conversation_dict['createdAt']
            
        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]['role'] == "user":
            createdMessageValue =  cosmos_conversation_client.create_message(
                uuid=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1]
            )
            if createdMessageValue == "Conversation not found":
                raise Exception("Conversation not found for the given conversation ID: " + conversation_id + ".")
        else:
            raise Exception("No user message found")
        
        cosmos_conversation_client.cosmosdb_client.close()
        
        # Submit request to Chat Completions for response
        request_body = req.get_json()
        history_metadata['conversation_id'] = conversation_id
        request_body['history_metadata'] = history_metadata
        return conversation_internal(request_body)
       
    except Exception as e:
        logging.exception("Exception in /history/generate")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)


@app.route("history/update", methods=["POST"])
def update_conversation(req: func.HttpRequest) -> func.HttpResponse:
    authenticated_user = get_authenticated_user_details(request_headers=req.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    request_json = req.get_json()
    conversation_id = request_json.get('conversation_id', None)

    try:
        # make sure cosmos is configured
        cosmos_conversation_client = init_cosmosdb_client()
        if not cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        if not conversation_id:
            raise Exception("No conversation_id found")
            
        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]['role'] == "assistant":
            if len(messages) > 1 and messages[-2].get('role', None) == "tool":
                # write the tool message first
                cosmos_conversation_client.create_message(
                    uuid=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    user_id=user_id,
                    input_message=messages[-2]
                )
            # write the assistant message
                cosmos_conversation_client.create_message(
                uuid=messages[-1]['id'],
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1]
            )
        else:
            raise Exception("No bot messages found")
        
        # Submit request to Chat Completions for response
        cosmos_conversation_client.cosmosdb_client.close()
        response = {'success': True}
        return func.HttpResponse(json.dumps(response), 200)
       
       
    except Exception as e:
        logging.exception("Exception in /history/update")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

@app.route("history/message_feedback", methods=["POST"])
def update_message(req: func.HttpRequest) -> func.HttpResponse:
    authenticated_user = get_authenticated_user_details(request_headers=req.headers)
    user_id = authenticated_user['user_principal_id']
    cosmos_conversation_client = init_cosmosdb_client()

    ## check request for message_id
    request_json = req.get_json()
    message_id = request_json.get('message_id', None)
    message_feedback = request_json.get("message_feedback", None)
    try:
        if not message_id:
            return func.HttpResponse(json.dumps({"error": "message_id is required"}), 400)
        
        if not message_feedback:
            return func.HttpResponse(json.dumps({"error": "message_feedback is required"}), 400)
        
        ## update the message in cosmos
        updated_message =  cosmos_conversation_client.update_message_feedback(user_id, message_id, message_feedback)
        if updated_message:
            return func.HttpResponse(json.dumps({"message": f"Successfully updated message with feedback {message_feedback}", "message_id": message_id}), 200)
        else:
            return func.HttpResponse(json.dumps({"error": f"Unable to update message {message_id}. It either does not exist or the user does not have access to it."}), 404)
        
    except Exception as e:
        logging.exception("Exception in /history/message_feedback")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)


@app.route("history/delete", methods=["DELETE"])
def delete_conversation(req: func.HttpRequest) -> func.HttpResponse:
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=req.headers)
    user_id = authenticated_user['user_principal_id']
    
    ## check request for conversation_id
    request_json = req.get_json()
    conversation_id = request_json.get('conversation_id', None)

    try: 
        if not conversation_id:
            return func.HttpResponse(json.dumps({"error": "conversation_id is required"}), 400)
        
        ## make sure cosmos is configured
        cosmos_conversation_client = init_cosmosdb_client()
        if not cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos first
        deleted_messages = cosmos_conversation_client.delete_messages(conversation_id, user_id)

        ## Now delete the conversation 
        deleted_conversation = cosmos_conversation_client.delete_conversation(user_id, conversation_id)

        cosmos_conversation_client.cosmosdb_client.close()

        return func.HttpResponse(json.dumps({"message": "Successfully deleted conversation and messages", "conversation_id": conversation_id}), 200)
    except Exception as e:
        logging.exception("Exception in /history/delete")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)


@app.route("history/list", methods=["GET"])
def list_conversations(req: func.HttpRequest) -> func.HttpResponse:
    offset = req.args.get("offset", 0)
    authenticated_user = get_authenticated_user_details(request_headers=req.headers)
    user_id = authenticated_user['user_principal_id']

    ## make sure cosmos is configured
    cosmos_conversation_client = init_cosmosdb_client()
    if not cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversations from cosmos
    conversations = cosmos_conversation_client.get_conversations(user_id, offset=offset, limit=25)
    cosmos_conversation_client.cosmosdb_client.close()
    if not isinstance(conversations, list):
        return func.HttpResponse(json.dumps({"error": f"No conversations for {user_id} were found"}), 404)

    ## return the conversation ids

    return func.HttpResponse(json.dumps(conversations), 200)


@app.route("history/read", methods=["POST"])
def get_conversation(req: func.HttpRequest) -> func.HttpResponse:
    authenticated_user = get_authenticated_user_details(request_headers=req.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    request_json = req.get_json()
    conversation_id = request_json.get('conversation_id', None)
    
    if not conversation_id:
        return func.HttpResponse(json.dumps({"error": "conversation_id is required"}), 400)
    
    ## make sure cosmos is configured
    cosmos_conversation_client = init_cosmosdb_client()
    if not cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation object and the related messages from cosmos
    conversation = cosmos_conversation_client.get_conversation(user_id, conversation_id)
    ## return the conversation id and the messages in the bot frontend format
    if not conversation:
        return func.HttpResponse(json.dumps({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404)
    
    # get the messages for the conversation from cosmos
    conversation_messages = cosmos_conversation_client.get_messages(user_id, conversation_id)

    ## format the messages in the bot frontend format
    messages = [{'id': msg['id'], 'role': msg['role'], 'content': msg['content'], 'createdAt': msg['createdAt'], 'feedback': msg.get('feedback')} for msg in conversation_messages]

    cosmos_conversation_client.cosmosdb_client.close()
    return func.HttpResponse(json.dumps({"conversation_id": conversation_id, "messages": messages}), 200)

@app.route("history/rename", methods=["POST"])
def rename_conversation(req: func.HttpRequest) -> func.HttpResponse:
    authenticated_user = get_authenticated_user_details(request_headers=req.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    request_json = req.get_json()
    conversation_id = request_json.get('conversation_id', None)
    
    if not conversation_id:
        return func.HttpResponse(json.dumps({"error": "conversation_id is required"}), 400)
    
    ## make sure cosmos is configured
    cosmos_conversation_client = init_cosmosdb_client()
    if not cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")
    
    ## get the conversation from cosmos
    conversation = cosmos_conversation_client.get_conversation(user_id, conversation_id)
    if not conversation:
        return func.HttpResponse(json.dumps({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404)

    ## update the title
    title = request_json.get("title", None)
    if not title:
        return func.HttpResponse(json.dumps({"error": "title is required"}), 400)
    conversation['title'] = title
    updated_conversation = cosmos_conversation_client.upsert_conversation(conversation)

    cosmos_conversation_client.cosmosdb_client.close()
    return func.HttpResponse(json.dumps(updated_conversation), 200)

@app.route("history/delete_all", methods=["DELETE"])
def delete_all_conversations(req: func.HttpRequest) -> func.HttpResponse:
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=req.headers)
    user_id = authenticated_user['user_principal_id']

    # get conversations for user
    try:
        ## make sure cosmos is configured
        cosmos_conversation_client = init_cosmosdb_client()
        if not cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        conversations = cosmos_conversation_client.get_conversations(user_id, offset=0, limit=None)
        if not conversations:
            return func.HttpResponse(json.dumps({"error": f"No conversations for {user_id} were found"}), 404)
        
        # delete each conversation
        for conversation in conversations:
            ## delete the conversation messages from cosmos first
            deleted_messages = cosmos_conversation_client.delete_messages(conversation['id'], user_id)

            ## Now delete the conversation 
            deleted_conversation = cosmos_conversation_client.delete_conversation(user_id, conversation['id'])
        cosmos_conversation_client.cosmosdb_client.close()
        return func.HttpResponse(json.dumps({"message": f"Successfully deleted conversation and messages for user {user_id}"}), 200)
    
    except Exception as e:
        logging.exception("Exception in /history/delete_all")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)

@app.route("history/clear", methods=["POST"])
def clear_messages(req: func.HttpRequest) -> func.HttpResponse:
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=req.headers)
    user_id = authenticated_user['user_principal_id']
    
    ## check request for conversation_id
    request_json = req.get_json()
    
    conversation_id = request_json.get('conversation_id', None)

    try: 
        if not conversation_id:
            return func.HttpResponse(json.dumps({"error": "conversation_id is required"}), 400)
        
        ## make sure cosmos is configured
        cosmos_conversation_client = init_cosmosdb_client()
        if not cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos
        deleted_messages = cosmos_conversation_client.delete_messages(conversation_id, user_id)

        return func.HttpResponse(json.dumps({"message": "Successfully deleted messages in conversation", "conversation_id": conversation_id}), 200)
    except Exception as e:
        logging.exception("Exception in /history/clear_messages")
        return func.HttpResponse(json.dumps({"error": str(e)}), 500)


@app.route("history/ensure", methods=["GET"])
def ensure_cosmos(req: func.HttpRequest) -> func.HttpResponse:
    if not AZURE_COSMOSDB_ACCOUNT:
        return func.HttpResponse(json.dumps({"error": "CosmosDB is not configured"}), 404)
    
    try:
        cosmos_conversation_client = init_cosmosdb_client()
        success, err = cosmos_conversation_client.ensure()
        if not cosmos_conversation_client or not success:
            if err:
                return func.HttpResponse(json.dumps({"error": err}), 422)
            return func.HttpResponse(json.dumps({"error": "CosmosDB is not configured or not working"}), 500)
        
        cosmos_conversation_client.cosmosdb_client.close()
        return func.HttpResponse(json.dumps({"message": "CosmosDB is configured and working"}), 200)
    except Exception as e:
        logging.exception("Exception in /history/ensure")
        cosmos_exception = str(e)
        if "Invalid credentials" in cosmos_exception:
            return func.HttpResponse(json.dumps({"error": cosmos_exception}), 401)
        elif "Invalid CosmosDB database name" in cosmos_exception:
            return func.HttpResponse(json.dumps({"error": f"{cosmos_exception} {AZURE_COSMOSDB_DATABASE} for account {AZURE_COSMOSDB_ACCOUNT}"}), 422)
        elif "Invalid CosmosDB container name" in cosmos_exception:
            return func.HttpResponse(json.dumps({"error": f"{cosmos_exception}: {AZURE_COSMOSDB_CONVERSATIONS_CONTAINER}"}), 422)
        else:
            return func.HttpResponse(json.dumps({"error": "CosmosDB is not working"}), 500)


def generate_title(conversation_messages):
    ## make sure the messages are sorted by _ts descending
    title_prompt = 'Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Respond with a json object in the format {{"title": string}}. Do not include any other commentary or description.'

    messages = [{'role': msg['role'], 'content': msg['content']} for msg in conversation_messages]
    messages.append({'role': 'user', 'content': title_prompt})

    try:
        azure_openai_client = init_openai_client(use_data=False)
        response = azure_openai_client.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=messages,
            temperature=1,
            max_tokens=64
        )
        
        title = json.loads(response.choices[0].message.content)['title']
        return title
    except Exception as e:
        return messages[-2]['content']

