import streamlit as st
import fitz
import re
from google import genai
from google.api_core import retry
from google.genai import types
import kagglehub
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import chroma_db
import chromadb
from chromadb import EmbeddingFunction, Embeddings

# Initialise Gemini client & retry wrapper
genai_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
        genai.models.Models.generate_content
    )

def load_chroma_db():
    class GeminiEmbeddingFunction(EmbeddingFunction):
        doc_mode = True

        def __call__(self, texts) -> Embeddings:
            task = "retrieval_document" if self.doc_mode else "retrieval_query"
            resp = genai_client.models.embed_content(
                model="models/text-embedding-004",
                contents=texts,
                config=types.EmbedContentConfig(task_type=task),
            )
            return [e.values for e in resp.embeddings]
    client = chromadb.PersistentClient(path="./chroma_db")  # your folder
    embed_fn = GeminiEmbeddingFunction()
    collection = client.get_collection(
        name="fundamentals_of_electric_circuits",
        embedding_function=embed_fn
    )
    return collection, embed_fn

vector_store, embed_fn = load_chroma_db()


# Streamlit UI
st.title("AI-Powered Assistant for *Fundamentals of Electric Circuits*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask a question about the textbook:")

develepor_info = {
    "Name": "Hafiz Syed Sharjeel Najam",
    "Education": "BE - Electrical Engineering from NED University Karachi",
    "Intrests": ['Power Systems', 'Generative AI', 'Machine Learning', 'Data Science', 'Python'],
    "Certifications": {
        "Certified Data Scientist": "NED Academy",
        "Electric Power Systems": "University at Buffalo",
        "Google 5 Days Generative AI Intensive": "Google & Kaggle",
        "Machine Learning": "Kaggle",
        "Python": "Kaggle"
    },
    "Contact": {
        "Linkedin": "https://pk.linkedin.com/in/syedsharjeel321",
        "Github": "https://github.com/syed-sharjeel",
        "Email": "syedsharjeel321@gmail.com",
        "Kaggle": "https://kaggle.com/syedsharjeelnajam"
    }
}

if query:
    embed_fn.doc_mode = False
    query_embed = embed_fn([query])[0]
    results = vector_store.query    (
        query_embeddings=[query_embed], n_results=4, include=["documents"]
    )
    context = "\n\n".join(results["documents"][0])

    prompt = f"""
    You are a helpful tutor for first-year electrical-engineering students.
    Use the reference text below to answer clearly and concisely in plain language.
    Reference text (from the book): {context}. If {query} is irrelevant to electrical concept you may ignore it. If someone asks about develepor
    answer him from {develepor_info}

    QUESTION: {query}
    """

    answer = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    ).text

    st.session_state.messages += [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]

    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(answer)
