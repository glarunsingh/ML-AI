import os
import logging
import pickle
import faiss
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setting Environment variable
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://dskumar.openai.azure.com/'
os.environ["AZURE_OPENAI_API_KEY"] = "62855d6dd08945819bf83aee0c104127"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "DskumarDeployment"
os.environ['OPENAI_TYPE'] = "Azure"
os.environ["LLM_MODEL"] = "gpt-35-turbo-16k"
os.environ["LLM_EMBEDDING_MODEL"] = "dskumar-text-embedding-ada-002"

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["10K_docs", "process_docs", "HR_docs"] = Field(
        ..., description="Given a user question choose which datasource would be most relevant for answering their question",
    )

# LLM with function call
azurechatmodel = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)
embeddings = AzureOpenAIEmbeddings(model=os.environ["LLM_EMBEDDING_MODEL"])

structured_llm = azurechatmodel.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to the appropriate data source.
Based on the document type the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Helper function to load or create FAISS vector store
def get_or_create_faiss_index(loader, embeddings, index_path, metadata_path):
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        try:
            index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                index_to_docstore_id = metadata['index_to_docstore_id']
                docstore = metadata['docstore']
            vector = FAISS(
                index=index,
                embedding_function=embeddings,
                index_to_docstore_id=index_to_docstore_id,
                docstore=docstore
            )
            print(f"Loaded FAISS index and metadata from {index_path} and {metadata_path}.")
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading FAISS index or metadata: {e}")
            print("Recreating index and metadata...")
            docs = loader.load()
            documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
            vector = FAISS.from_documents(documents, embeddings)
            faiss.write_index(vector.index, index_path)
            metadata = {
                'index_to_docstore_id': vector.index_to_docstore_id,
                'docstore': vector.docstore
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
    else:
        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        faiss.write_index(vector.index, index_path)
        metadata = {
            'index_to_docstore_id': vector.index_to_docstore_id,
            'docstore': vector.docstore
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Created and saved FAISS index and metadata to {index_path} and {metadata_path}.")
    return vector

# Datasource paths
finance_index_path = 'faiss_finance_index.index'
finance_metadata_path = 'faiss_finance_metadata.pkl'
process_index_path = 'faiss_process_index.index'
process_metadata_path = 'faiss_process_metadata.pkl'
hr_index_path = 'faiss_hr_index.index'
hr_metadata_path = 'faiss_hr_metadata.pkl'

# Datasources
finance_loader = PyPDFLoader(r".\citi-2022-annual-report.pdf")
finance_vector = get_or_create_faiss_index(finance_loader, embeddings, finance_index_path, finance_metadata_path)
tenk_retriever = finance_vector.as_retriever()

process_loader = PyPDFLoader(r".\The_Barclays_Way.pdf")
process_vector = get_or_create_faiss_index(process_loader, embeddings, process_index_path, process_metadata_path)
process_retriever = process_vector.as_retriever()

hr_loader = PyPDFLoader(r".\Citibank-Online.pdf")
hr_vector = get_or_create_faiss_index(hr_loader, embeddings, hr_index_path, hr_metadata_path)
hr_retriever = hr_vector.as_retriever()

# Define router
router = prompt | structured_llm

question = """what is the profit for citi bank last year?"""

result = router.invoke({"question": question})

print(result.datasource)

def choose_route(result):
    if "10K_docs" in result.datasource:
        return tenk_retriever
    elif "process_docs" in result.datasource:
        return process_retriever
    else:
        return hr_retriever

retriever = choose_route(result)

# Function to query Azure OpenAI
def query_azureopenai(prompt, temperature=0.7, max_tokens=150):
    response = azurechatmodel.invoke(prompt)
    return response.content

# Function to query the retriever and then use Azure OpenAI to generate a response
def query_retriever_with_azureopenai(query, retriever):
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = query_azureopenai(prompt)
    return response

# Example of querying the retriever and using Azure OpenAI to generate an answer
response = query_retriever_with_azureopenai(question, retriever)
print(response)
