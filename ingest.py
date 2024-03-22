from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate 
from langchain.llms import CTransformers
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

DATA_PATH="documents/"
DB_FAISS_PATH="vectorstores/db_faiss"


def create_vectordb():
    try:
        # Ensure the directory exists or create it
       

        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        print(db.index.ntotal)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_vectordb()
    


# def final_result(query):
#     qa_result=qa_bot()
#     response=qa_result({'query':query})
#     return response


# import streamlit as st



# def main():
#     st.title("LawLens⚖️")
#     st.write("Start with your first Question.......")
   
#     st.sidebar.title("What's LawLens all About?")
#     st.sidebar.image("co.jpg",use_column_width=True)
#     st.sidebar.write("👩‍⚖️LawLens is Trained on the Indian Constitution to help people better understand their rights and laws.👨‍⚖️ ")

#     query = st.text_input("Enter your question:")
    
#     if st.button("Submit"):
#         if query:
#             response = final_result(query)
#             st.write("Response:", response)
#         else:
#             st.warning("Please enter a question.")

#     st.image("123.jpg",width=200)

# if __name__ == "__main__":
#     main()

