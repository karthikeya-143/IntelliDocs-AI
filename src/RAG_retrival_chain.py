from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectordb, llm):
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """
You are an assistant. Answer the question **only using the information provided in the context below**.
Do not use any outside knowledge.

Context:
{context}

Question:
{question}

Instructions:
- If the answer can be found in the context, provide it concisely.
- If the answer is not present in the context, reply exactly: "I don't know."
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain