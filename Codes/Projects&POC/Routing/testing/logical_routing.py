import os
import logging

# Setting Environment variable
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://dskumar.openai.azure.com/'
os.environ["AZURE_OPENAI_API_KEY"] = "62855d6dd08945819bf83aee0c104127"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "DskumarDeployment"
os.environ['OPENAI_TYPE'] = "Azure"
os.environ["LLM_MODEL"] = "gpt-35-turbo-16k"
os.environ["LLM_EMBEDDING_MODEL"] = "dskumar-text-embedding-ada-002"

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["10K_docs", "process_docs", "HR_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

# LLM with function call 
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

#datasources
#10K document 
finance_loader = PyPDFLoader(r".\citi-2022-annual-report.pdf")
# Load documents and create vector store
finance_docs = finance_loader.load()
finance_documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(finance_docs)
vector = FAISS.from_documents(finance_documents, embeddings)
tenk_retriever = vector.as_retriever()

#process management
process_loader = PyPDFLoader(r".\The_Barclays_Way.pdf")
# Load documents and create vector store
process_docs = process_loader.load()
process_documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(process_docs)
vector = FAISS.from_documents(process_documents, embeddings)
process_retriever = vector.as_retriever()

#HR docs
HR_loader = PyPDFLoader(r".\Citibank-Online.pdf")
# Load documents and create vector store
HR_docs = HR_loader.load()
HR_documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(HR_docs)
vector = FAISS.from_documents(HR_documents, embeddings)
HR_retriever = vector.as_retriever()

# Define router 
router = prompt | structured_llm

question = """What the annual profit of citi bank last year?

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("english")
"""

result = router.invoke({"question": question})

print(result.datasource)


def choose_route(result):
    if "10K_docs" in result.datasource:
        ### Logic here 
        return tenk_retriever
    elif "process_docs" in result.datasource:
        ### Logic here 
        return process_retriever
    else:
        ### Logic here 
        return HR_retriever

result = choose_route(result)

# Function to query Azure OpenAI
def query_azureopenai(prompt, temperature=0.7, max_tokens=150):
    response = azurechatmodel.invoke(prompt)
    #response = AzureChatOpenAI.Completion.create(engine=model,prompt=prompt,temperature=temperature,max_tokens=max_tokens)
    return response.content

# Function to query the retriever and then use Azure OpenAI to generate a response
def query_retriever_with_azureopenai(query, retriever):
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = query_azureopenai(prompt)
    return response

# Example of querying the retriever and using Azure OpenAI to generate an answer
#question = "How is Barclays plays its role in society?"
#print(choose_route)
response = query_retriever_with_azureopenai(question, result)
print(response)


#from langchain_core.runnables import RunnableLambda
#full_chain = router | RunnableLambda(choose_route)
#full_chain.invoke({"question": question})
