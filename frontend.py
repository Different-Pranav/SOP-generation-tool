import streamlit as st
import requests

# API URL
API_URL = "http://localhost:8000/generate-sop/"  # Update if the API is hosted elsewhere

# Streamlit App
st.title("SOP Generation Platform")
st.subheader("Generate your personalized Statement of Purpose")

# Input form
with st.form("sop_form"):
    st.header("Student Information")
    name = st.text_input("Full Name", placeholder="Enter your full name")
    background = st.text_area("Academic Background", placeholder="Summarize your academic journey")
    gpa = st.text_input("GPA", placeholder="Enter your GPA")
    work_experience = st.text_area("Work Experience (Optional)", placeholder="List relevant work experience")
    achievements = st.text_area("Achievements (Optional)", placeholder="List notable achievements, separated by commas")
    background_story = st.text_area("Personal Background Story", placeholder="Describe your personal background story")
    goals = st.text_area("Goals", placeholder="Explain your career and academic goals")
    
    st.header("University and Program Details")
    university = st.text_input("University Name", placeholder="Enter the university name")
    program = st.text_input("Program Name", placeholder="Enter the program name")
    
    # Submit button
    submitted = st.form_submit_button("Generate SOP")
    
if submitted:
    # Validate input
    if not name or not background or not gpa or not background_story or not goals or not university or not program:
        st.error("Please fill in all required fields!")
    else:
        # Prepare payload
        payload = {
            "student_info": {
                "name": name,
                "background": background,
                "gpa": gpa,
                "work_experience": work_experience if work_experience else None,
                "achievements": achievements.split(",") if achievements else None,
                "background_story": background_story,
                "goals": goals,
            },
            "university": university,
            "program": program,
        }

        # Call the API
        with st.spinner("Generating your SOP..."):
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
                sop = response.json().get("sop", "Error: No SOP returned by the API")
                st.success("SOP generated successfully!")
                st.text_area("Generated SOP", sop, height=400)
            except requests.exceptions.RequestException as e:
                st.error(f"Error generating SOP: {e}")
