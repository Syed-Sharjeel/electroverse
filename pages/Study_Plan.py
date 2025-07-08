import streamlit as st
import fitz
import re
from google import genai
from google.api_core import retry
from google.genai import types
import kagglehub
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

CHROMA_DB_DIR = "chroma_db_study_plan" 
DB_name = 'syllabi_of_courses'

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
class GeminiEmbeddingFunction(EmbeddingFunction):
    doc_mode = True
    def __call__(self, input) -> Embeddings:
        task = "retrieval_document" if self.doc_mode else "retrieval_query"
        response = genai_client.models.embed_content(
            model = 'models/text-embedding-004',
            contents = input,
            config = types.EmbedContentConfig(
                task_type = task
            )
        )
        return [e.values for e in response.embeddings]

embed_fn = GeminiEmbeddingFunction()
embed_fn.doc_mode = True

collection = chroma_client.get_or_create_collection(
    name=DB_name,
    embedding_function=embed_fn
)


# Streamlit UI
import streamlit as st
st.header('Study Plan')

course = st.selectbox('Enter Course', options=['EE-125 - Basic Electrical Engineering',
                                      'EE-126 - Circuit Analysis',
                                      'EE-156 - Engineering Drawing',
                                      'EE-163 - Compupter & Programming',
                                      'EE-223 -  Instrumentation and Measurement',
                                      'EE-232 - Signals and Systems',
                                      'EE-264 -  Data Structures and Algorithms',
                                      'EE-282 -  Electromagnetic Fields',
                                      'EE-346 - Electrical Machines I',
                                      'EE-347 - Electrical Machines II',
                                      'EE-354 - Embedded Systems',
                                      'EE-352 - Electrical Power Transmission',
                                      'EE-359 -  Electrical Power Distribution and Utilization',
                                      'EE-362 -  Power System Analysis',
                                      'EE 375 - Feedback Control Systems',
                                      'EE 313 - Power Electronics',
                                      'EE-414 - Power Generation',
                                      'EE-457 - Electrical Power System Protection'])
chapter = st.text_input("Enter Topic or write 'All Chapters' for whole course Plan")
number_of_days = st.number_input('Number of Days', value=None)
if st.button('Generate Plan'):
    query = f'Generate {number_of_days} days study plan for course {course} topic {chapter}'

    if query:
        embed_fn.doc_mode = False
        query_embed = embed_fn([query])[0]
        results = collection.query    (
            query_embeddings=[query_embed], n_results=4, include=["documents"]
        )
        context = "\n\n".join(results["documents"][0])

        prompt = f"""
        You are a helpful guider for electrical-engineering students guiding about desired study plan depending on number of days, course, and topic.
        Use the reference text below to answer clearly and concisely in plain language. Generate well formated result with bulleted points for better readability.
        Reference text (from the book): {collection}. If the course is out of domain you msy ignore it.
        Just give the study plan with days/weeks breakout nothing more then that.
        For Example:
        Query: Generate 5 days study plan for course EE-125 topic 'Capacitors and Inductors'.
        EE-125: Capacitors and Inductors - 5 days study plan for course EE-125 topic capacitors and inductors
        Expected Output:

Day 1:

Introduction to Capacitors: Definition, symbol, units (Farads), physical construction, and basic function (storing charge).
Capacitance Calculation: Factors affecting capacitance (plate area, distance, dielectric), permittivity.
Capacitor Voltage-Current Relationship: Understanding i = C(dv/dt).
Capacitor Energy Storage: Derivation and application of the formula E = (1/2)CV^2.
Simple Capacitor Circuits: Series and parallel combinations of capacitors and equivalent capacitance calculation.
Day 2:

Introduction to Inductors: Definition, symbol, units (Henries), physical construction (coil), and basic function (storing energy in a magnetic field).
Inductance Calculation: Factors affecting inductance (number of turns, core material, geometry).
Inductor Voltage-Current Relationship: Understanding v = L(di/dt).
Inductor Energy Storage: Derivation and application of the formula E = (1/2)LI^2.
Simple Inductor Circuits: Series and parallel combinations of inductors and equivalent inductance calculation.
Day 3:

Capacitor and Inductor Behavior in DC Circuits:
Capacitors: Open circuit at steady-state DC.
Inductors: Short circuit at steady-state DC.
RC Circuits: Charging and discharging of capacitors, time constant (τ = RC), transient response, voltage and current waveforms.
RL Circuits: Current rise and decay in inductors, time constant (τ = L/R), transient response, voltage and current waveforms.
Day 4:

Source-Free RC and RL Circuits: Derivation of equations for natural response.
Step Response of RC and RL Circuits: Analyzing the response to a sudden change in voltage or current.
Practical Applications: Examples of capacitors and inductors in filtering, energy storage, and timing circuits.
Day 5:

Review: Go over all the material from days 1-4.
Problem Solving: Practice a variety of problems related to capacitors, inductors, RC circuits, and RL circuits. Focus on understanding the concepts and applying the formulas.
Quiz: Self-test to check understanding and identify weak areas.
        QUESTION: {query}
        """

        answer = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        ).text
        st.write(answer)
