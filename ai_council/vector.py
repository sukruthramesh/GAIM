from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ai_council.constants import *

###############################################################################
# Name : check_files_folders
# Function : lists documents and checks if there is a vector 
# Returns : list of files and boolean flagging existent of vector db
###############################################################################
def check_files_folders():
    files = os.listdir(DATA_DOC_FOLDER)
    vectors = True if len(os.listdir(VECTOR_DB_FOLDER)) else False
    return files, vectors

###############################################################################
# Name : get_vector_db
# Function : Retreives the db or creates a db if outdates 
# Returns : vector stores
###############################################################################
def get_vector_db():
    if verify_file_vectorisation():
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vs = FAISS.load_local(VECTOR_DB_FOLDER, embeddings=embeddings, allow_dangerous_deserialization=True)
        return vs
    else:
        print("Creating the vector DB")
        vs = create_vector_db()
        return vs

###############################################################################
# Name : create_vector_db
# Function : Creates and saves a vector db of all the files in the Docs folder 
# Returns : vector stores
###############################################################################
def create_vector_db():
    files, vectors = check_files_folders()
    documents = [DATA_DOC_FOLDER+"/"+k for k in files]
    pages = []
    vectorised_documents = []

    for document in documents:
        reader = PdfReader(document)
        for i, page in enumerate(reader.pages, start=1):
            txt = (page.extract_text() or "").strip()
            if len(txt.split()) < 50 or "table of contents" in txt.lower():
                continue
            pages.append({"text": txt, "source": document, "page": i})
        vectorised_documents.append(document)

    docs = [Document(page_content=p["text"], metadata={"source": p["source"], "page": p["page"]})
        for p in pages]
    # Chunking (keep it modest so it’s fast)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vs = FAISS.from_texts(
        texts=[c.page_content for c in chunks],
        embedding=embeddings,
        metadatas=[c.metadata for c in chunks],
    )
    vs.save_local(VECTOR_DB_FOLDER)
    files_pd = pd.DataFrame(files)
    files_pd.to_csv('ai_council/Vectorised_files.csv', index=False)
    return vs

###############################################################################
# Name : verify_file_vectorisation
# Function : Checks if all the files have been vectorised
# Returns : Boolean that flags non existent or outdated vectors
###############################################################################
def verify_file_vectorisation():
    try:
        files, vectors = check_files_folders()
        vectoried_files = pd.read_csv('ai_council/Vectorised_files.csv').to_numpy()
        same_files = sorted(files) == sorted(vectoried_files)
        return same_files and vectors
    except Exception as e:
        print(e)
        return False