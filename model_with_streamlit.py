import chainlit as cl
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers

DB_FAISS_PATH = "C://Users//HP//Desktop//Devcation//vectorstores//db_faiss"




def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores.
    """
    custom_prompt_template = """Use the following pieces of information to answer the user's question. If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only returns the helpful answer below and nothing else.
    Helpful answer:"""

    # Ensure 'context' is included in the input variables
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    return prompt


def load_llm():
    llm = CTransformers(model="llama-2-7b.ggmlv3.q8_0.bin",
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.5)
    return llm


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=False,  # Change to False to exclude sources
        chain_type_kwargs={'prompt': prompt})
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

import streamlit as st

def final_result(query):
    qa_result = qa_bot()  # Initialize the QA bot
    response = qa_result({'query': query})  # Get response from the QA bot
    return response['result']  # Extract the result from the response

def main():
    # CSS styles
    main_style = """
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    """
    title_style = """
        font-size: 36px;
        font-weight: bold;
        color: #333333;
    """
    sidebar_style = """
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    """

    # Main content
    st.title("LawLens⚖️")
    st.markdown("<p style='{}'>Start with your first question...</p>".format(title_style), unsafe_allow_html=True)
    st.image("123.jpg",width=200)
    # Sidebar
    st.sidebar.markdown("<p style='{}'>What's LawLens all About?</p>".format(title_style), unsafe_allow_html=True)
    st.sidebar.image("co.jpg", use_column_width=True)
    st.sidebar.markdown("<p style='{}'>👩‍⚖️ LawLens is Trained on the Indian Constitution to help people better understand their rights and laws. 👨‍⚖️</p>".format(sidebar_style), unsafe_allow_html=True)

    query = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        if query:
            response = final_result(query)
            st.write("Response:", response)
        else:
            st.warning("Please enter a question.")

   

if __name__ == "__main__":
    main()
