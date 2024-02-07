from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

persist_directory = './db'

def main():
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))

    documents = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

    db = Chroma.from_documents(texts, embeddings, persist_directory='db')

    db.persist()
    db = None

if __name__ == "__main__":
    main()
