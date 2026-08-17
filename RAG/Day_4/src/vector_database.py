
# vectordatabase _- - - -- > Chroma
# Embeddings - - - -> Model - - - - - -> Huggingface

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.settings import settings

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

def create_vectordb(texts, metadatas):
    vectorstore = Chroma.from_texts(
        texts= texts,
        metadatas=metadatas,
        embedding= embeddings,
        collection_name = settings.collection_name,
        persist_directory= settings.persist_directory
    )
    return vectorstore

def load_vectordb():
    vectorstore = Chroma(
        embedding_function= embeddings,
        collection_name = settings.collection_name,
        persist_directory= settings.persist_directory
    )
    return vectorstore





