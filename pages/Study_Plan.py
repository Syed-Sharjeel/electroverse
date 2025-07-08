import streamlit as st
import fitz
import re
from google import genai
from google.api_core import retry
from google.genai import types

def initial_run_gemini():
    genai_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
    if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
        genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
            genai.models.Models.generate_content
        )
    return genai_client

def doc_run():
    doc_desc = fitz.open(r'pages/syllabi_ee.txt')
    pages = [page.get_text() for page in doc_desc]
    course_blocks_desc = re.split(r"(EE-\d{3}.*?)\n", "\n".join(pages))[1:]  # Splits on course code
    course_outlines_desc = {
        f"{course_desc.strip()}": desc.strip()
        for course_desc, desc in zip(course_blocks_desc[0::2], course_blocks_desc[1::2])
    }
    return course_outlines_desc


st.title('Study Plan')
genai_client = initial_run_gemini()
course_outlines_desc = doc_run()

course = st.selectbox('Select Course', options = course_outlines_desc.keys())
topic = st.text_input('Enter Topic', placeholder="Enter 'All Topics' for Whole Course")
days = st.number_input("Enter Number of Days", min_value=1)
syllabi = course_outlines_desc[course]
if st.button('Generate Plan'):
    query = f"Generate {days} days plan for {topic} topic for course {course}"
    prompt = f"""You are a helpful, intelligent, and concise **Study Plan Generator Machine** for electrical engineering students.
    Your job is to generate a personalized study plan for a given course, topic, and number of days. Syllabus for each course is {syllabi}. The plan should break the content into **daily learning goals**, with clear, concise points and no unnecessary details.
    ### Instructions:
    - Format the output as a clean, readable plan.
    - Use bullet points or numbered days (Day 1, Day 2, etc.).
    - If the user selects “All Chapters” or doesn’t specify a topic, generate a full course breakdown.
    - Only use concepts relevant to the course and topic.
    - Don’t add filler text or explanations—just generate the study plan directly.
    - The plan must be balanced across the given number of days.
    ### Example Query:
    Generate a 5-day study plan for course EE-125, topic “Capacitors and Inductors”.
    ### Example Output:
    Day 1:
    - Introduction to Capacitors
    - Capacitance principles and formulas
    - Energy stored in a capacitor
    Day 2:
    - Introduction to Inductors
    - Inductance concepts and calculation
    - Energy stored in an inductor
    Day 3:
    - Capacitors and Inductors in DC circuits
    - Time constant and transient analysis (RC, RL)
    Day 4:
    - Source-free and step responses
    - Practical use cases
    Day 5:
    - Review and practice problems
    ### Query: Generate a {days}-day study plan for course {course}, topic: {topic}."""

    answer = genai_client.models.generate_content(
        model = "gemini-2.0-flash",
        contents = prompt
    ).text

    st.write(answer)
