import os
import openai
import sys
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import datetime
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

sys.path.append('../..')
openai.api_key  = os.environ['OPENAI_API_KEY']

def process_input(input):
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )


        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        embedding = OpenAIEmbeddings()

        directory='chroma/'

        # loader = PyPDFLoader("yashthaker.pdf")
        loader = TextLoader('data.txt')
        pages = loader.load()

        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        after_splitting = r_splitter.split_documents(pages)

        vectordb = Chroma.from_documents(
            documents=after_splitting,
            embedding=embedding,
            persist_directory=directory
        )

        retriever=vectordb.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
        )

        result = qa({"question": input})

        return result


def main():
    st.title('Chat with resume')
    question = st.text_input('Ask a question')
    if not question:
        st.warning("please enter the question")
        return

    if st.button("Submit"):
        result = process_input(question)
        if result:
             st.subheader("Answer")
             st.write(result["answer"])
        else:
             st.write("Unknown error")

if __name__ == "__main__":
    main()