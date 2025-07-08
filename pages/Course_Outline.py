import streamlit as st
import fitz
from google import genai
from google.api_core import retry
from google.genai import types
import re

def initial_run_gemini():
    genai_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
    if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
        genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
            genai.models.Models.generate_content
        )
    return genai_client

def doc_run():
    doc_desc = fitz.open(r'C:\Users\Syed_Sharjeel\Desktop\SS\electroverse\pages\syllabi_ee.txt')
    pages = [page.get_text() for page in doc_desc]
    course_blocks_desc = re.split(r"(EE-\d{3}.*?)\n", "\n".join(pages))[1:]  # Splits on course code
    course_outlines_desc = {
        f"{course_desc.strip()}": desc.strip()
        for course_desc, desc in zip(course_blocks_desc[0::2], course_blocks_desc[1::2])
    }
    return course_outlines_desc


st.title('Course Outline')
genai_client = initial_run_gemini()
course_outlines_desc = doc_run()

course = st.selectbox('Select Course', options = course_outlines_desc.keys())
if st.button('Search'):
    text = course_outlines_desc[course]
    prompt = f"""Italicize the Headings with bulleted points and remove page numbers for better readability from {text}. Avoid such things: 'Here's the formatted content with italicized headings, bullet points, and removed page numbers:'"""
    answer = genai_client.models.generate_content(
        model = "gemini-2.0-flash",
        contents = prompt
    ).text
    st.write(answer)
