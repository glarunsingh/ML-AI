import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchain_openai import AzureChatOpenAI ## LLM model - Chat GPT 3
from langchain_openai import AzureOpenAIEmbeddings##embedding type
from langchain_community.vectorstores import FAISS ## Vector DB
from langchain.retrievers import SVMRetriever ### classification model
from langchain.chains import QAGenerationChain ### QA pair Generator 
from langchain.text_splitter import RecursiveCharacterTextSplitter ### for spliting the doc
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from llmlingua import PromptCompressor
import pandas as pd


 
#Setting Environment variable
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://dskumar.openai.azure.com/'
os.environ["AZURE_OPENAI_API_KEY"] ="62855d6dd08945819bf83aee0c104127"
os.environ["DEPLOYMENT_NAME"] ="DskumarDeployment"
os.environ['OPENAI_TYPE']="Azure"
os.environ["LLM_MODEL"] = "gpt-35-turbo-16k"
os.environ["LLM_EMBEDDING_MODEL"] = "dskumar-text-embedding-ada-002"


st.set_page_config(page_title="PDF and Text Analyzer",page_icon=':clipboard:')

@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            # Get the total number of pages in the PDF
            total_pages = len(pdf_reader.pages)
            st.info(f"Total Pages: {total_pages}")

           
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num    ]
                text += page.extract_text()
              
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text

@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    st.info("`Retriving ...`")
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
            retriever = vectorstore.as_retriever(k=5)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)
    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits
    
    st.info("`Splitting doc ...`")  
    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits

@st.cache_data
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list
    
    
    st.info("`Generating sample questions ...`")
    n = len(text)
    
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    ##chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    chain = QAGenerationChain.from_llm(AzureChatOpenAI(deployment_name=os.getenv("DEPLOYMENT_NAME"),
                                                       openai_api_type='Azure',))
   
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
           
        except:
            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full


# ...

def main():
    
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
        <p></a></p>
    </div>
    """

    st.markdown(foot, unsafe_allow_html=True)
    
    # Add custom CSS
    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
            
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image("C:\\Users\\thara\\my_streamlit_app\\pdf-analysis-using-streamlit-main\\img\\logo2.png")


    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">PDF Analyzer - AzureChatOpenAI</h1>
        <sup style="margin-left:5px;font-size:small; color: green;"></sup>
    </div>
    """,
    unsafe_allow_html=True,
        )
    
    st.sidebar.title("Menu")
    
    embedding_option = st.sidebar.radio(
        "Choose Embeddings", ["Azure OpenAI Embeddings"])

    
    retriever_type = st.sidebar.selectbox(
        "Choose Retriever", ["SIMILARITY SEARCH", "SUPPORT VECTOR MACHINES"])

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

   
    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")
        
        ################  Azure OpenAI snippet
        # Create an instance of Azure OpenAI
        st.write("Interacting with Azure Open AI........................")
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("DEPLOYMENT_NAME"),openai_api_type='Azure',
        )

        # Embed using OpenAI embeddings
            # Embed using OpenAI embeddings or HuggingFace embeddings
        if embedding_option == "Azure OpenAI Embeddings":
            embeddings = AzureOpenAIEmbeddings(model = os.environ["LLM_EMBEDDING_MODEL"],)
        elif embedding_option == "HuggingFace Embeddings(slower)":
            # Replace "bert-base-uncased" with the desired HuggingFace model
            embeddings = HuggingFaceEmbeddings()

        retriever = create_retriever(embeddings, splits, retriever_type)

        # Initialize the RetrievalQA chain with streaming output, Use this If Streaming = True
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])
          

        # Check if there are no generated question-answer pairs in the session state
        if 'eval_set' not in st.session_state:
            # Use the generate_eval function to generate question-answer pairs
            num_eval_questions = 0 # Number of question-answer pairs to generate
            st.session_state.eval_set = generate_eval(
                loaded_text, num_eval_questions, 3000)
            
            
         #Write QA into CSV file
        df = pd.DataFrame(columns=['Question', 'Without_Compress_Answer', 'With_Compress_Answer'])
        
        st.write("Storing QA in Dataframe")
         
        for i, qa_pair in enumerate(st.session_state.eval_set):
             new_row= {'Question': qa_pair['question'], 'Without_Compress_Answer': qa_pair['answer'], 'With_Compress_Answer': ''}
             df.loc[len(df)] = new_row
             
             
        # Specify the file path to save the CSV file
        file_path = 'C:\\Users\\thara\\Downloads\\CSV_Path\\demo.csv'
        

        # Write the DataFrame to a CSV file
        df.to_csv(file_path, index=False)    
        
        st.write("CSV file is generated")

       # Display the question-answer pairs in the sidebar with smaller text
        for i, qa_pair in enumerate(st.session_state.eval_set):
            st.sidebar.markdown(
                f"""
                <div class="css-card">
                <span class="card-tag">Question {i + 1}</span>
                    <p style="font-size: 12px;color: blue;">{qa_pair['question']}</p>
                    <p style="font-size: 12px;color: blue;">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # <h4 style="font-size: 14px;color: blue;">Question {i + 1}:</h4>
            # <h4 style="font-size: 14px;color: blue;">Answer {i + 1}:</h4>
        st.write("Ready to answer questions.")

        # Question and answering
        user_question = st.text_input("Enter your question:")
        if user_question:
         
        
              context_0= retriever.get_relevant_documents(user_question)
              ##st.write(f"context_0:{context_0}")
            
              
              concatenated_content = ""
              for doc in context_0:
                  concatenated_content += doc.page_content + "\n"
                  
              context_1 = [concatenated_content]
             
              
             
              
              # Create lingua object
              llm_lingua = PromptCompressor(
                  model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                  device_map='cpu',
                  use_llmlingua2=True,
              )

              
              ##
              compressed_prompt = llm_lingua.compress_prompt(
                  context_1,
                  rate=0.8,
                  force_tokens=["!", ".", "?", "\n"],
                  #force_tokens=["?", "\n"],
                  chunk_end_tokens=[".", "\n"],
                  return_word_label=True,
                  drop_consecutive=True,
                  ##target_token=300,
              )
              
              
              prompt = "\n\n".join([compressed_prompt["compressed_prompt"],user_question])

              message = [{"role": "user", "content": prompt},]
                         #{"role": "system", "content": "You are a helpful assistant designed to output exactly with in the context.Do not output extra words."},]

              request_data = {
            ##"max_tokens": 50,
              "temperature": 0,
              "top_p": 1,
              "n": 1,
              "stream": False,}
            
             
              message = HumanMessage(prompt)
             ## llm_Response_Compression = llm([message],temperature=0,top_p=1, n=1,stream=False,)
              llm_Response_Compression = llm([message],**request_data)
              st.write("compressed Answer:",llm_Response_Compression.content)
              st.write("prompt_tokens:",llm_Response_Compression.response_metadata['token_usage']['prompt_tokens'])
              st.write("completion_tokens:",llm_Response_Compression.response_metadata['token_usage']['completion_tokens'])
              st.write("total_tokens:",llm_Response_Compression.response_metadata['token_usage']['total_tokens'])
              
              ###################### without Compression
              
              source =concatenated_content ##string object
              prompt_1 = "\n\n".join([source, user_question])
              message_1 = [{"role": "user", "content": prompt_1},]

             
              message_1 = HumanMessage(prompt_1)
              ##max_tokens": 100
              llm_Response_without_Compression = llm([message_1],**request_data)
              st.write("without compression Answer:",llm_Response_without_Compression.content)
              st.write("prompt_tokens:",llm_Response_without_Compression.response_metadata['token_usage']['prompt_tokens'])
              st.write("completion_tokens:",llm_Response_without_Compression.response_metadata['token_usage']['completion_tokens'])
              st.write("total_tokens:",llm_Response_without_Compression.response_metadata['token_usage']['total_tokens'])
              
                 
if __name__ == "__main__":
    main()
          
             