from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os

# Setting Environment variable
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://dskumar.openai.azure.com/'
os.environ["AZURE_OPENAI_API_KEY"] = "62855d6dd08945819bf83aee0c104127"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "DskumarDeployment"
os.environ['OPENAI_TYPE'] = "Azure"
os.environ["LLM_MODEL"] = "gpt-35-turbo-16k"
os.environ["LLM_EMBEDDING_MODEL"] = "dskumar-text-embedding-ada-002"

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(model=os.environ["LLM_EMBEDDING_MODEL"])

from langchain_community.document_loaders.pdf import PyPDFLoader
loader = PyPDFLoader(r".\citi-2022-annual-report.pdf")
output = loader.load()
print(output[0].page_content)

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents and create vector store
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
tenk_retriever = vector.as_retriever()

# LLM with function call
azurechatmodel = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)

# Two prompts
template_10K = """You are very smart at reading financial statements of a company. \
You are great at answering questions about 10K documents in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

process_template = """You are very good at process management. You are great at answering process management questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

# Embed prompts
prompt_templates = [template_10K, process_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Route question to prompt
def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input['query'])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Check if the question is about the 10K document
    if most_similar == template_10K:
        relevant_docs = tenk_retriever.retrieve(input["query"])
        input["context"] = "\n".join(doc.page_content for doc in relevant_docs)
    else:
        input["context"] = ""
    print("Using template_10K" if most_similar == template_10K else "Using process")
    return PromptTemplate.from_template(most_similar)

chain = (
    {"query": RunnablePassthrough(), "context": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | azurechatmodel
    | StrOutputParser()
)

print(chain.invoke({"query": "summarize the annual report of Citi bank from the given document?"}))
