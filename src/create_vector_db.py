from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

def create_vector_db(text, embedder):
    doc=Document(page_content=text)
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    docs=splitter.split_documents([doc])
    print(docs)
    vectordb=FAISS.from_documents(docs, embedding=embedder)
    vectordb.save_local("research_paper_vector_db")
    return vectordb