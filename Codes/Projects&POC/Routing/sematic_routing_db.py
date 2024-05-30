import os
import logging
import pickle
from typing import List

# Setting Environment variable
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://dskumar.openai.azure.com/'
os.environ["AZURE_OPENAI_API_KEY"] = "62855d6dd08945819bf83aee0c104127"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "DskumarDeployment"
os.environ['OPENAI_TYPE'] = "Azure"
os.environ["LLM_MODEL"] = "gpt-35-turbo-16k"
os.environ["LLM_EMBEDDING_MODEL"] = "dskumar-text-embedding-ada-002"

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths
pdf_paths = {
    'finance': r".\citi-2022-annual-report.pdf",
    'hr': r".\The_Barclays_Way.pdf",
    'process_management': r".\Citibank-Online.pdf"
}

vector_store_path = 'vector_store.pkl'

# Function to create vector store from PDFs
def create_vector_store(pdf_paths: dict) -> FAISS:
    all_documents = []
    for department, pdf_path in pdf_paths.items():
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        all_documents.extend(chunks)

    embeddings = AzureOpenAIEmbeddings(model=os.environ["LLM_EMBEDDING_MODEL"])
    embedded_documents = embeddings.embed_documents([doc.page_content for doc in all_documents])

    vector_store = FAISS()
    vector_store.add_texts(embedded_documents)
    
    return vector_store

# Load or create vector store
if os.path.exists(vector_store_path):
    with open(vector_store_path, 'rb') as f:
        vector_store = pickle.load(f)
else:
    vector_store = create_vector_store(pdf_paths)
    with open(vector_store_path, 'wb') as f:
        pickle.dump(vector_store, f)

# Rest of your routing and question-answering logic here

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Two prompts
finance_template = """You are a very smart financial reporter/advisor in reading financial statements. \
You are great at answering questions about financial statements in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

process_template = """You are a very good at process management. You are great at answering process management questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

# Embed prompts
# LLM with function call
azurechatmodel = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)
embeddings = AzureOpenAIEmbeddings(model=os.environ["LLM_EMBEDDING_MODEL"])
prompt_templates = [finance_template, process_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Route question to prompt 
def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt 
    print("Using process" if most_similar == process_template else "Using finance")
    return PromptTemplate.from_template(most_similar)

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | azurechatmodel
    | StrOutputParser()
)

print(chain.invoke("how we can manage the malpractice performed by the employees?"))