import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
from constants import CHROMA_SETTINGS

## Function to save Uploaded file in local directory 
def saveuploadedfile(uploadedfile):
    with open(os.path.join('data',uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved file {} to directory".format(uploadedfile.name))



checkpoint = "LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,device_map="auto",torch_dtype = torch.float32
)


@st.cache_resource
def llm_pipeline():
    pipeline = pipeline(
        'text2text-generation',model=base_model,tokenizer=tokenizer,max_length=256,do_sample=True,temperature=0.3,top_p=0.95
    )
    local_llm= HuggingFacePipeline(pipeline=pipeline)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline
    embeddings = SentenceTransformerEmbeddings(model_name='all-Mini-L6-v2')
    db = Chroma(persist_directory="db",embedding_function=embeddings,client_settings=CHROMA_SETTINGS)
    retriever = db.as_reteriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,chain_type="stuff",retriever=retriever,return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response=''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer,generated_text

## Setting page configuration 
def main():
    st.set_page_config(layout="wide")

    st.title("Pdf App")

    uploaded_pdf = st.file_uploader("Upload PDF: ",type=['pdf'])

    if uploaded_pdf is not None:
        col1,col2 = st.columns([1,1])

        with col1:
            input_file = saveuploadedfile(uploaded_pdf)
            pdf_file = "data/"+uploaded_pdf.name
        with col2:
            question = st.text_area("Enter your Question ")
            if st.button("Search"):
                st.info("your question: "+question)
                st.info("Your Answer")
                answer,metadata = process_answer(question)
                st.write(answer)
                st.write(metadata)

if __name__ == '__main__':
    main()


