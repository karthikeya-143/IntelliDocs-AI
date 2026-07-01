import json
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.load_and_extract_text import extract_pdf_sections, extract_text_from_pdf
from src.detect_and_split_sections import refine_sections,split_sections_with_content

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
llm_model=os.getenv("LLM_MODEL")
embedding_model=os.getenv("EMBEDDING_MODEL")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
) 
# print(llm.invoke("how is Baahubali?"))
embedder = HuggingFaceEmbeddings(model_name=embedding_model)

if __name__ == "__main__":
    extract_text=extract_text_from_pdf("research-paper.pdf")
    # print(extract_text)
    extracted_sections=extract_pdf_sections(full_text=extract_text)
    # with open("extracted_sections.json","w") as f:
    #     json.dump(extracted_sections,f,indent=4)

    refined_sections=refine_sections(extracted_sections,llm)
    # with open("refined_sections.json","w") as f:
    #     json.dump(refined_sections,f,indent=4)
    sections_with_content=split_sections_with_content(extract_text,refined_sections)
    with open("sections_with_content.json","w") as f:
        json.dump(sections_with_content,f,indent=4)