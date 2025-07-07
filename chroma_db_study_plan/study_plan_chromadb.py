import chromadb
import streamlit as st
import fitz
import re
from google import genai
from google.api_core import retry
from google.genai import types
import kagglehub
from chromadb import EmbeddingFunction, Embeddings
from chromadb.config import Settings

# Initialise Gemini client & retry wrapper
genai_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
        genai.models.Models.generate_content
    )

doc = r'C:\Users\Syed_Sharjeel\Desktop\SS\electroverse\pages\courses_eed.pdf'
import fitz
doc = fitz.open(doc)
all_pages = [page.get_text() for page in doc]
doc.close()

# Define path where DB files should be saved (like SQLite)
CHROMA_DB_DIR = "chroma_db_study_plan"  # you can use './chroma_db_store' or any path
DB_name = 'syllabi_of_courses'

# Create Chroma persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
class GeminiEmbeddingFunction(EmbeddingFunction):
    doc_mode = True
    def __call__(self, input:doc) -> Embeddings:
        task = "retrieval_document" if self.doc_mode else "retrieval_query"
        response = genai_client.models.embed_content(
            model = 'models/text-embedding-004',
            contents = input,
            config = types.EmbedContentConfig(
                task_type = task
            )
        )
        return [e.values for e in response.embeddings]
# Custom embedding function (your Gemini-based embedding)
embed_fn = GeminiEmbeddingFunction()
embed_fn.doc_mode = True

# Create or get the collection
collection = chroma_client.get_or_create_collection(
    name=DB_name,
    embedding_function=embed_fn
)
