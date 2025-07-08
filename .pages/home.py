import streamlit as st
st.set_page_config(page_title="ElectroVerse", layout="wide")
st.title('Welcome to ElectroVerse')
st.header('Your AI Assistant for Fundamentals of Electric Cicruits')
st.write(f"""ElectroVerse is a smart, student-focused AI assistant designed to simplify and personalize your 
        learning experience in Electrical Engineering, starting with the widely-used textbook "Fundamentals of Electric Circuits" by Alexander & Sadiku.
        Whether you're preparing for an exam, reviewing a chapter, or planning your study schedule — ElectroVerse is here to help!""")
st.markdown('---')
st.title('Key Features')
st.header('1. Smart Q&A Chatbot')
st.write(f"""Ask any question related to Fundamentals of Electric Circuits and get accurate, context-aware answers
        powered by RAG (Retrieval-Augmented Generation). The chatbot searches real textbook content, ensuring
        reliable and course-specific responses.""")

st.header('2. Study Plan Generator')
st.write(f"""Just input your Course Code, Chapter/Topic, and the number of Preparation Days, and ElectroVerse will
        automatically generate a personalized study plan to help you stay on track.""")

st.header('3. Course Outline Viewer')
st.write(f"""Confused about what’s in a course? Simply select the course code to instantly view a well-structured 
        outline.""")

st.header('4. Practice Questions')
st.write(f"""ElectroVerse will soon allow you to generate practice questions based on your input topic, chapter,
        or course – for smarter, targeted revision.""")

st.title('Why ElectroVerse?')
st.write(f"""- Tailored for Engineering students, especially those studying Electrical & Electronics.
- Grounded answers directly from authentic textbook content.
- Easy-to-use interface powered by Streamlit.
- Built with cutting-edge Generative AI & RAG technologies.""")

st.header("Start exploring ElectroVerse – and power up your learning!")
