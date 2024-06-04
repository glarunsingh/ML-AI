import os
import pickle
import faiss
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.utils.math import cosine_similarity

# Setting Environment variable
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://dskumar.openai.azure.com/'
os.environ["AZURE_OPENAI_API_KEY"] = "62855d6dd08945819bf83aee0c104127"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "DskumarDeployment"
os.environ['OPENAI_TYPE'] = "Azure"
os.environ["LLM_MODEL"] = "gpt-35-turbo-16k"
os.environ["LLM_EMBEDDING_MODEL"] = "dskumar-text-embedding-ada-002"

# Two prompts
finance_template = """You are a very smart financial reporter/advisor in reading financial statements. \
You are great at answering questions about financial statements in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}

Context:
{context}"""

process_template = """You are a very good at process management. You are great at answering process management questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}

Context:
{context}"""

# Embed prompts
azurechatmodel = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)
embeddings = AzureOpenAIEmbeddings(model=os.environ["LLM_EMBEDDING_MODEL"])
prompt_templates = [finance_template, process_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

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
finance_retriever = finance_vector.as_retriever()

process_loader = PyPDFLoader(r".\The_Barclays_Way.pdf")
process_vector = get_or_create_faiss_index(process_loader, embeddings, process_index_path, process_metadata_path)
process_retriever = process_vector.as_retriever()

hr_loader = PyPDFLoader(r".\Citibank-Online.pdf")
hr_vector = get_or_create_faiss_index(hr_loader, embeddings, hr_index_path, hr_metadata_path)
hr_retriever = hr_vector.as_retriever()

# Route question to prompt
def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar_index = similarity.argmax()
    most_similar = prompt_templates[most_similar_index]
    chosen_route = "process" if most_similar_index == 1 else "finance"
    # Chosen prompt
    print(f"Using {chosen_route} route")
    return chosen_route, PromptTemplate.from_template(most_similar)

# Function to retrieve relevant data and pass to LLM
def get_response(input):
    chosen_route, result_template = prompt_router(input)
    
    if chosen_route == "finance":
        retriever = finance_retriever
    elif chosen_route == "process":
        retriever = process_retriever
    else:
        retriever = hr_retriever

    # Retrieve relevant data
    docs = retriever.invoke(input["query"])
    context = " ".join([doc.page_content for doc in docs])

    # Format prompt with retrieved context
    formatted_prompt = result_template.format(query=input["query"], context=context)

    # Prepare the message for the LLM
    messages = [
        {
            "role": "system",
            "content": formatted_prompt
        }
    ]

    # Get response from LLM
    response = azurechatmodel.invoke(messages)
    return response

# Example usage
input_query = {"query": "explain big bang theory?"}
response = get_response(input_query)
print(f"Response: {response.content}")