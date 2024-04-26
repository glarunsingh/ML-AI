# Importing the required packages

import numpy as np
import streamlit as st
import os
import hashlib
from shutil import move
from scipy.spatial import distance

from sentence_transformers import SentenceTransformer
import PyPDF2
import docx2txt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import pickle
from ironpdf import *

import pandas as pd

import time

# Define your column names
columns = ['Moved File', 'Comparison File Name', "Comparison Method", 'Similarity Score']

# Create an empty DataFrame with these columns
df = pd.DataFrame(columns=columns)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
img_model = VGG16(weights='imagenet', include_top=False)
img_ext = (".jpg", ".jpeg", ".tiff", ".png")
allowed_filetype = (".jpg", ".jpeg", ".tiff", ".png", ".pdf", ".docx")

# Set document text similarity score
cosine_score = 0.9

# Set Image similarity score
img_similarity_score = 0.95

# Set Image similarity score
pdf_img_similarity_score = 0.76

document_folder = ""
duplicates_folder = ""
st.set_page_config(layout="wide", page_title="Duplicate files remover", page_icon=":newspaper:")

# Initialize a variable to store the handle to the last displayed message
last_info = None

# Defining the SHA Algorithm
def create_hash_database(folder_path):
    global last_info
    time.sleep(2)
    if last_info is not None:
        last_info.empty()
    last_info=st.info("No Hash Database found!  \nCreating New Hash Database")
    hash_database = {}  # creating a dictionary for hash_database
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'rb') as f:
                    md5_hash = hashlib.md5(f.read()).hexdigest()
                    sha256_hash = hashlib.sha256(f.read()).hexdigest()
                hash_database[filename] = (md5_hash, sha256_hash)

            except FileNotFoundError:
                st.info(f"Error: File '{filename}' not found. Hence skipping...")
    time.sleep(2)
    if last_info is not None:
        last_info.empty()
    last_info=st.info("Hash Database Created")
    time.sleep(2)
    return hash_database


# Create a folder if  folder doesn't exist
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Check for duplicate in the file
def check_duplicate(filename, hash_database):
    if filename in hash_database:
        return True

    return False


def move_duplicate_llm(filename, duplicate_file, similarity_score, source_folder, destination_folder):
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    if os.path.exists(source_path):
        try:
            move(source_path, destination_path)
            new_row_data = {'Moved File': filename, 'Comparison File Name': duplicate_file, "Comparison Method": "LLM",
                            'Similarity Score': min(round(similarity_score * 100, 2), 100)}
            df.loc[len(df)] = new_row_data
            # st.info(
            #     f"Duplicate '{filename}' moved to duplicate folder with similarity score of {similarity_score} with file {duplicate_file}")

        except Exception as e:
            st.info(f"Error moving '{filename}': {e}")


def move_duplicate(filename, source_folder, destination_folder, duplicate_filename):
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    if os.path.exists(source_path):
        try:
            move(source_path, destination_path)
            new_row_data = {'Moved File': filename, 'Comparison File Name': duplicate_filename,
                            "Comparison Method": "HASH",
                            'Similarity Score': 100}
            df.loc[len(df)] = new_row_data
            # st.info(f"Duplicate '{filename}' moved to duplicate folder as it is duplicate to {duplicate_filename}")

        except Exception as e:
            st.info(f"Error moving '{filename}': {e}")


def extract_text(path):
    text = ""
    if path.endswith('.docx'):
        text = docx2txt.process(path)

    elif path.endswith('.pdf'):
        with open(path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    else:
        pass

    return text


def cosine_similarity_img(img_1_embd, img_2_embd):
    cosine_similarity = 1 - distance.cosine(img_1_embd, img_2_embd)
    return cosine_similarity


# Generate Image Embedding
def img_embedding(path):
    # Preprocess image
    img = image.load_img(path, target_size=(224, 224))  # Resize to match VGG16 input size
    img_array = image.img_to_array(img)
    img_2d = np.expand_dims(img_array, axis=0)
    img_processed = preprocess_input(img_2d)
    img_feature = img_model.predict(img_processed, verbose=0)

    # Flatten the image to get embedding
    img_embd = img_feature.flatten()

    return img_embd


def convert_pdf2img(filename, file_path):
    pdf = PdfDocument.FromFile(file_path)
    create_folder(os.path.join(document_folder, "pdf2img"))
    out_path = os.path.join(document_folder, "pdf2img", filename + ".png")
    pdf.RasterizeToImageFiles(out_path, DPI=96)
    return out_path


def create_embd_database(folder_path):
    global last_info
    time.sleep(3)
    if last_info is not None:
        last_info.empty()
    last_info=st.info("No Embedding Database Found!  \nCreating New Embedding Database")

    embd_database = {}  # creating a dictionary for embd_database
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):

            # Files are either pdf or docx
            try:
                if filename.endswith((".pdf", ".docx")):

                    text = extract_text(file_path)

                    # if text in the pdf is null then change it to the img file
                    if text == "" and filename.endswith(".pdf"):
                        pdf2img_path = convert_pdf2img(filename, file_path)

                        # Generate image embedding
                        embedding = img_embedding(pdf2img_path)

                    else:
                        embedding = model.encode(text, normalize_embeddings=True)

                # Files are images
                elif filename.endswith(img_ext):
                    # Generate image embedding
                    embedding = img_embedding(file_path)

                else:
                    pass

                # Store embedding in dictionary
                embd_database[filename] = embedding

            except FileNotFoundError:
                st.info(f"Error: File '{filename}' not found. Hence skipping...")

            except Exception as e:
                st.info(f"Error moving '{filename}': {e}")
    time.sleep(3)
    if last_info is not None:
        last_info.empty()
    last_info=st.info("Embedding Database Created")
    time.sleep(2)


    return embd_database


def sha_algo():
    global last_info
    if last_info is not None:
        last_info.empty()
    last_info = st.info('Hash model is running....')

    create_folder(duplicates_folder)  # Create a duplicate folder if doesn't exists
    # load existing hash database or create a new one
    if os.path.exists("hash_database.txt"):
        with open("hash_database.txt", "r") as f:
            hash_database = eval(f.read())
    else:
        hash_database = create_hash_database(document_folder)
        with open("hash_database.txt", "w") as f:
            f.write(str(hash_database))

    # Check for new documents
    for filename in os.listdir(document_folder):
        file_path = os.path.join(document_folder, filename)
        if os.path.isfile(file_path) and filename.endswith(allowed_filetype):
            # New document: update hash database and potentially move existing duplicates
            try:
                with open(file_path, "rb") as f:
                    md5_hash = hashlib.md5(f.read()).hexdigest()
                    sha256_hash = hashlib.sha256(f.read()).hexdigest()
                hash_database[filename] = (md5_hash, sha256_hash)

                # check for existing duplicates based on MD5 or SHA-256
                for existing_filename, existing_hashes in hash_database.items():
                    if (existing_filename != filename) and (md5_hash == existing_hashes[0]):  # checking for MD5
                        # move existing duplicates
                        move_duplicate(existing_filename, document_folder, duplicates_folder, filename)
                        break  # only move one duplicate per new document
            except FileNotFoundError:
                st.info(f"Error: File'{filename}' not found. Skipping...")

    with open("hash_database.txt", "w") as f:
        f.write(str(hash_database))

    time.sleep(3)
    if last_info is not None:
        last_info.empty()

    last_info = st.info("Hash Model Completed")

def llm_algo():
    global last_info
    time.sleep(2)
    if last_info is not None:
        last_info.empty()
    last_info = st.info('\nLLM model is running....\n')


    create_folder(duplicates_folder)  # Create a duplicate folder if doesn't exists

    # Setting up the model bge-large-zh-v1.5
    if os.path.exists("embd_database.pkl"):
        with open("embd_database.pkl", "rb") as file:
            embd_database = pickle.load(file)
    else:

        embd_database = create_embd_database(
            document_folder)  # Duplicate folder path is provided because at the database if any duplicates are there the function will move the files to duplicates folder
        with open("embd_database.pkl", "wb") as file:
            pickle.dump(embd_database, file)

    for filename in os.listdir(document_folder):
        file_path = os.path.join(document_folder, filename)

        if os.path.isfile(file_path) and filename.endswith(allowed_filetype):

            # Check for new pdf or docx files
            try:
                if filename.endswith((".pdf", ".docx")):
                    text = extract_text(file_path)
                    # if text in the pdf is null then change it to the img file
                    if text == "" and filename.endswith(".pdf"):
                        pdf2img_path = convert_pdf2img(filename, file_path)

                        # Generate image embedding
                        embedding = img_embedding(pdf2img_path)

                    else:
                        embedding = model.encode(text, normalize_embeddings=True)

                # Check for image files
                elif filename.endswith(img_ext):
                    # Generate image embedding
                    embedding = img_embedding(file_path)

                else:
                    pass

                # store embedding in the dictionary
                embd_database[filename] = embedding

                for existing_filename, existing_embd in embd_database.items():
                    if existing_filename != filename:
                        # checking for Duplicates
                        if existing_embd.shape[0] == embedding.shape[0] == 384:  # and .endswith((".pdf",".docx"))
                            similarity_score = existing_embd @ embedding.T

                            if similarity_score > cosine_score:  # Check with the text threshold limit(cosine score)
                                move_duplicate_llm(existing_filename, filename, similarity_score, document_folder,
                                                   duplicates_folder)
                                # break #only move one duplicate per new document

                        elif existing_embd.shape[0] == embedding.shape[0] == 25088:
                            similarity_score_img = cosine_similarity_img(embedding, existing_embd)
                            if (existing_filename.endswith(".pdf")) ^ (
                                    filename.endswith(".pdf")):  # Only one of them should be pdf(XOR condition)
                                if similarity_score_img > pdf_img_similarity_score:  # Check with the image threshold limit(cosine score)
                                    move_duplicate_llm(existing_filename, filename, similarity_score_img,
                                                       document_folder, duplicates_folder)
                                # break

                            elif similarity_score_img > img_similarity_score:  # Check with the image threshold limit(cosine score)
                                move_duplicate_llm(existing_filename, filename, similarity_score_img, document_folder,
                                                   duplicates_folder)
                                # break
                            else:
                                pass
                        else:
                            pass

            except FileNotFoundError:
                st.info(f"Error: File'{filename}' not found. Skipping...")

            except Exception as e:
                st.info(f"Error moving '{filename}': {e}")

    # Save the embedding dictionary in pkl
    with open("embd_database.pkl", "wb") as file:
        pickle.dump(embd_database, file)


    time.sleep(2)
    if last_info is not None:
        last_info.empty()
    last_info = st.info("LLM Model completed")
    time.sleep(3)
    if last_info is not None:
        last_info.empty()



def main():
    global document_folder
    global duplicates_folder

    # Define directory paths using streamlit
    document_folder = st.text_input('Input folder path where documents are stored')
    duplicates_folder = st.text_input('Input folder path where duplicate documents file to be stored')
    if document_folder == "" or duplicates_folder == "":
        st.text("Enter the folders path")
    if document_folder != "":
        if os.path.isdir(document_folder):
            st.text(f"Selected document folder: {document_folder}")
        else:
            st.text("No path exists! \nEnter the documents folder again")
    if duplicates_folder != "":
        if os.path.isdir(document_folder):
            st.text(f"Selected duplicate document folder: {duplicates_folder}")
        else:
            st.text("No path exists! \nEnter the duplicate documents folder again")

    # document_folder = r"C:\Users\ankit\Downloads\Project Duplicate\Testing docs"
    # duplicates_folder = r"C:\Users\ankit\Downloads\Project Duplicate\Testing docs\Duplicate_files"
    # sha_algo()
    # llm_algo()

    if os.path.isdir(document_folder) and os.path.isdir(duplicates_folder) and st.button("Move Duplicate files"):
        sha_algo()
        llm_algo()
        st.success("\n Completed!")
        if len(df) != 0:
            st.info("Please Find the summary")
            st.dataframe(df, use_container_width=True)
        else:
            st.success("No Duplicates found")


if __name__ == "__main__":
    main()
