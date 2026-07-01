import json
import os
from flask import Flask,render_template,request,jsonify
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.load_and_extract_text import extract_pdf_sections, extract_text_from_pdf
from src.detect_and_split_sections import refine_sections,split_sections_with_content

load_dotenv()

app=Flask(__name__)
app.config['UPLOAD_FOLDER']='uploads'

os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)

groq_api_key=os.getenv("GROQ_API_KEY")
llm_model=os.getenv("LLM_MODEL")
embedding_model=os.getenv("EMBEDDING_MODEL")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
) 
# print(llm.invoke("how is Baahubali?"))
embedder = HuggingFaceEmbeddings(model_name=embedding_model)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)