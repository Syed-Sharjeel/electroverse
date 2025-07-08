import streamlit as st
import fitz
from google import genai
from google.api_core import retry
from google.genai import types
import chromadb
from chromadb import EmbeddingFunction, Embeddings

st.title('Course Outline')

# Initialize Gemini client & retry logic
genai_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
        genai.models.Models.generate_content
    )

class GeminiEmbeddingFunction(EmbeddingFunction):
    doc_mode = True
    def __call__(self, input) -> Embeddings:
        task = 'retrieval_document' if self.doc_mode else 'retrieval_query'
        response = genai_client.models.embed_content(
            model='models/text-embedding-004',
            contents=input,
            config=types.EmbedContentConfig(task_type=task)
        )
        return [e.values for e in response.embeddings]


DB_PATH = "chroma_course_outline"
DB_NAME = "course_outline"

embed_fn = GeminiEmbeddingFunction()
embed_fn.doc_mode = True

chroma_client = chromadb.PersistentClient(path=DB_PATH)

collection = chroma_client.get_or_create_collection(
    name=DB_NAME,
    embedding_function=embed_fn
)

course = st.selectbox('Enter Course', options=[
    'EE-125 - Basic Electrical Engineering',
    'EE-126 - Circuit Analysis',
    'EE-156 - Engineering Drawing',
    'EE-163 - Computer & Programming',
    'EE-223 - Instrumentation and Measurement',
    'EE-232 - Signals and Systems',
    'EE-264 - Data Structures and Algorithms',
    'EE-282 - Electromagnetic Fields',
    'EE-346 - Electrical Machines I',
    'EE-347 - Electrical Machines II',
    'EE-354 - Embedded Systems',
    'EE-352 - Electrical Power Transmission',
    'EE-359 - Electrical Power Distribution and Utilization',
    'EE-362 - Power System Analysis',
    'EE 375 - Feedback Control Systems',
    'EE 313 - Power Electronics',
    'EE-414 - Power Generation',
    'EE-457 - Electrical Power System Protection'
])

if st.button('Search'):
    query = f"Write down the Syllabi of Course {course}"
    prompt = f"""Just write down the Syllabi of courses of BE of course {course}. Use the reference text {collection}.
            Query: Write down syllabi of course EE-125
            Expected Output: EE-125 | Basic Electrical Engineering
            - *Fundamentals of Electric Circuits:* Charge, Current, Voltage and Power, Voltage and Current 
            Sources, Ohm’s Law. Equivalent resistance of a circuit. 
            - *Voltage and Current Laws:* Node, Loop and Branches, Kirchhoff’s Current Law (KCL), 
            Kirchhoff’s  Voltage  Law  (KVL),  single-loop  circuits,  single  Node  Pair  Circuit,  Series  and 
            Parallel Connected Independent Sources. 
            - *Circuit Analysis Techniques:* Nodal Analysis, Mesh Analysis, Linearity and Superposition, 
            Source Transformations, Thevenin and Norton Equivalent Circuits, Maximum Power Transfer 
            theorem. 
            - *Capacitors and Inductors:* Capacitor, Inductor, Inductance and Capacitance Combination, 
            voltage current relationship for inductor and capacitor. Energy storage. 
            - *Introduction  to  AC  Circuits:* Sinusoids  and  Phasors,  Phasor  Relationships  for  Circuit 
            Elements, Impedance and Admittance, Kirchhoff’s Laws in the Frequency Domain, Impedance 
            Combinations, Instantaneous and Average Power, Maximum Average Power Transfer, Effective 
            or RMS Value, Apparent Power and Power Factor, Complex Power, Conservation of AC Power. 
            - *Sinusoidal Steady-State Analysis:* Nodal Analysis, Mesh Analysis, Superposition Theorem, 
            Source Transformation, Thevenin and Norton Equivalent Circuits. 
            QUESTION: {query}"""
    

    answer = genai_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    ).text

    st.markdown(answer)
