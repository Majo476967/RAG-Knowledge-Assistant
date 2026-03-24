from langchain_community.vectorstores import FAISS

def build_vector_store(chunks, embedding_model):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore