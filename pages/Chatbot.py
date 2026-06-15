import streamlit as st
import chromadb
from chromadb import EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from groq import Groq

# CONFIGURATION

st.set_page_config(
    page_title="ElectroVerse",
    page_icon="⚡",
    layout="wide"
)

# GROQ CLIENT

groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

# EMBEDDING MODEL

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# CHROMA EMBEDDING FUNCTION

class BGEEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts) -> Embeddings:

        embeddings = embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

# LOAD CHROMA DB
@st.cache_resource
def load_chroma_db():
    embed_fn = BGEEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )
    collection = chroma_client.get_or_create_collection(
    name="fundamentals_of_electric_circuits"
)
    return collection, embed_fn
vector_store, embed_fn = load_chroma_db()

# STREAMLIT UI
st.title("⚡ ElectroVerse")
st.subheader("Fundamentals of Electric Circuits Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# USER INPUT
query = st.chat_input(
    "Ask a question about Electrical Engineering..."
)
if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    with st.chat_message("user"):
        st.markdown(query)

    # RETRIEVAL
    query_embedding = embed_fn([query])[0]
    results = vector_store.query(
        query_embeddings=[query_embedding],
        n_results=4,
        include=["documents"]
    )
    context = "\n\n".join(
        results["documents"][0]
    )

    prompt = f"""
You are ElectroVerse, an expert tutor for first-year Electrical Engineering students.

Instructions:
- Answer only using the provided context.
- Use simple language.
- Explain concepts step-by-step.
- If the answer is not found in the context, say:
  "The information is not available in the textbook."

Context:
{context}

Question:
{query}
"""
    # RESPONSE GENERATION
    with st.spinner("Thinking..."):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )
        answer = response.choices[0].message.content
    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
