import streamlit as st
import streamlit.components.v1 as components
st.title("ElectroVerse Feedback")
st.session_state.generate_form = False
if st.session_state.generate_form == False:
    device = st.radio("Choose your device type for optimal form view:", ["Desktop", "Mobile"])
    if st.button("Generate Form"):
        if device == "Desktop":
            width = 700
            height = 2010
        elif device == "Mobile":
            width = 300
            height = 2800 
        form_url = "https://docs.google.com/forms/d/e/1FAIpQLScNFCSlkJ6wGXZVqr36eZMrqj_z60ZJz5273z2VKFXWaY29gA/viewform"
        components.iframe(src=form_url, width=width, height=height)
