from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore
import logging

logging.getLogger().setLevel(logging.ERROR)
load_dotenv()

import pickle
with open("uploaded_files.pkl", 'rb') as f:
    uploaded_files = pickle.load(f)

from utils import get_pdf_text, split_docs
raw_text = get_pdf_text(uploaded_files)
text_chunks = split_docs(raw_text)

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"} #cpu or cuda
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
model_dimensions = 384

pc = Pinecone()
p_index = os.getenv('pinecone_index_name')
if p_index in pc.list_indexes().names():
    print("Deleted the Existing Index")
    pc.delete_index(p_index)

print("Creating the index")
pc.create_index(
    name=p_index,
    dimension=model_dimensions,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

namespace = "dhwaj_resume"
docsearch = PineconeVectorStore.from_texts(
    texts=text_chunks,
    index_name=p_index,
    embedding=hf, 
    namespace=namespace 
)

print("uploaded the data")

retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
)