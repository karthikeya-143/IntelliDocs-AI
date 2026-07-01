import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
llm_model=os.getenv("LLM_MODEL")
embedding_model=os.getenv("EMBEDDING_MODEL")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)
print(llm.invoke("how is Baahubali?"))
