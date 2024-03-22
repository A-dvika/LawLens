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


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot....")
    await msg.send()
    msg.content ="Hi, I'm LawLens, here to assist you with your rights, laws, and acts in India. How may I help you?"
    await msg.update()
    cl.user_session.set("chain", chain)



@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    if res["result"] is not None:  # Check if response is not None
        answer = res["result"]
        await cl.Message(content=answer).send()
    else:
        await cl.Message(content="I'm sorry, I don't have an answer for that.").send()
