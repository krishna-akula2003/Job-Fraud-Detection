# import streamlit as st
# import pandas as pd
# import joblib
# import sqlite3
# import re
# import string
# import unicodedata
# import pymupdf  # PyMuPDF for PDF processing
# from docx import Document  # python-docx for Word processing
# import os

# DB_PATH = "job_postings.db"
# MODEL_PATH = os.path.abspath("logistic_randomforest.pkl")

# def initialize_db():
#     with sqlite3.connect(DB_PATH) as conn:
#         conn.execute('''CREATE TABLE IF NOT EXISTS postings 
#                         (id INTEGER PRIMARY KEY, 
#                         job_title TEXT, 
#                         description TEXT, 
#                         prediction INTEGER)''')

# def fetch_stored_jobs():
#     with sqlite3.connect(DB_PATH) as conn:
#         df = pd.read_sql_query('SELECT job_title, description, prediction FROM postings', conn)
    
#     if not df.empty:
#         df = df.applymap(lambda x: unicodedata.normalize("NFKD", x).encode('utf-8', 'ignore').decode() if isinstance(x, str) else x)
    
#     return df

# def store_to_db(job_title, description, prediction):
#     with sqlite3.connect(DB_PATH) as conn:
#         conn.execute('INSERT INTO postings (job_title, description, prediction) VALUES (?, ?, ?)', 
#                      (job_title.encode('utf-8', 'ignore').decode(), 
#                       description.encode('utf-8', 'ignore').decode(), 
#                       prediction))

# def preprocess_text(text):
#     text = text.lower()
#     text = unicodedata.normalize("NFKD", text)
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def extract_text_from_pdf(file):
#     doc = pymupdf.open(stream=file.read(), filetype="pdf")
#     return "\n".join([page.get_text("text") for page in doc])

# def extract_text_from_docx(file):
#     doc = Document(file)
#     return "\n".join([para.text for para in doc.paragraphs])

# if os.path.exists(MODEL_PATH):
#     model = joblib.load(MODEL_PATH)
# else:
#     st.error(f"Model file not found at: {MODEL_PATH}")
#     st.stop()

# def predict_job_real_or_fake(description):
#     preprocessed_text = preprocess_text(description)
#     return model.predict([preprocessed_text])[0]

# def main():
#     st.set_page_config(page_title="Job Authenticity Checker", layout="wide")
#     st.markdown(
#         """
#         <style>
#             .main-title {
#                 font-size: 32px;
#                 font-weight: bold;
#                 text-align: center;
#                 color: #FFD700;
#             }
#             .sidebar .sidebar-content {
#                 background: linear-gradient(to bottom, #1e1e2e, #282a36);
#             }
#             .stButton>button {
#                 background: #4CAF50;
#                 color: white;
#                 font-size: 18px;
#                 border-radius: 8px;
#                 padding: 10px;
#                 width: 100%;
#             }
#             .stTextInput, .stTextArea {
#                 border-radius: 8px;
#                 border: 1px solid #ccc;
#             }
#             .prediction-box {
#                 text-align: center;
#                 font-size: 20px;
#                 font-weight: bold;
#                 padding: 15px;
#                 border-radius: 8px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     st.markdown('<p class="main-title">🔍 Job Posting Authenticity Checker</p>', unsafe_allow_html=True)
#     st.sidebar.title("Navigation")
    
#     menu = ["Home", "Stored Jobs", "About"]
#     choice = st.sidebar.radio("Go to", menu)
    
#     if choice == "Home":
#         st.subheader("Upload Job Details")
#         uploaded_file = st.file_uploader("Upload job description (PDF/DOCX)", type=["pdf", "docx"], key="file_uploader")
#         job_title, description = "", ""
        
#         if uploaded_file:
#             file_type = uploaded_file.type
#             if "pdf" in file_type:
#                 file_content = extract_text_from_pdf(uploaded_file)
#             elif "word" in file_type or "msword" in file_type:
#                 file_content = extract_text_from_docx(uploaded_file)

#             file_lines = file_content.split("\n", 1)
#             if len(file_lines) >= 2:
#                 job_title, description = file_lines[0], file_lines[1]
#             else:
#                 st.warning("Invalid file format. Ensure the first line is the job title.")
#                 return
#         else:
#             job_title = st.text_input("Job Title")
#             description = st.text_area("Job Description", height=200)
        
#         if st.button("Predict", use_container_width=True):
#             if job_title and description:
#                 prediction = predict_job_real_or_fake(description)
#                 result_text = "✅ Job Posting is Real" if prediction == 0 else "❌ Likely Fake"
#                 color = "#4CAF50" if prediction == 0 else "#FF5733"
#                 st.markdown(f'<p class="prediction-box" style="background-color:{color}; color:white;">{result_text}</p>', unsafe_allow_html=True)
#                 store_to_db(job_title, description, prediction)
#             else:
#                 st.warning("Please enter a job title and description.")
    
#     elif choice == "Stored Jobs":
#         st.subheader("Stored Job Postings")
#         df = fetch_stored_jobs()
#         if not df.empty:
#             st.dataframe(df, use_container_width=True)
#         else:
#             st.warning("No job postings stored yet.")
    
#     elif choice == "About":
#         st.subheader("About This App")
#         st.info("This app helps in detecting fake job postings using machine learning models.")

# if __name__ == "__main__":
#     initialize_db()
#     main()




# import streamlit as st
# import pandas as pd
# import joblib
# import sqlite3
# import re
# import string
# import unicodedata
# import pymupdf  # PyMuPDF for PDF processing
# from docx import Document  # python-docx for Word processing
# import os

# # ✅ Database & Model Paths
# DB_PATH = "job_postings.db"
# MODEL_PATH = "formvalidation/prediction/logistic_randomforest.pkl"
# VECTORIZER_PATH = "formvalidation/prediction/tfidf_vectorizer.pkl"

# # ✅ Initialize database
# def initialize_db():
#     with sqlite3.connect(DB_PATH) as conn:
#         conn.execute('''CREATE TABLE IF NOT EXISTS postings 
#                         (id INTEGER PRIMARY KEY, 
#                         job_title TEXT, 
#                         description TEXT, 
#                         prediction INTEGER)''')

# # ✅ Fetch stored jobs from the database
# def fetch_stored_jobs():
#     with sqlite3.connect(DB_PATH) as conn:
#         df = pd.read_sql_query('SELECT job_title, description, prediction FROM postings', conn)
    
#     if not df.empty:
#         df = df.applymap(lambda x: unicodedata.normalize("NFKD", x).encode('utf-8', 'ignore').decode() if isinstance(x, str) else x)
    
#     return df

# # ✅ Store new job predictions in database
# def store_to_db(job_title, description, prediction):
#     with sqlite3.connect(DB_PATH) as conn:
#         conn.execute('INSERT INTO postings (job_title, description, prediction) VALUES (?, ?, ?)', 
#                      (job_title.encode('utf-8', 'ignore').decode(), 
#                       description.encode('utf-8', 'ignore').decode(), 
#                       prediction))

# # ✅ Preprocessing function to clean text
# def preprocess_text(text):
#     text = text.lower()
#     text = unicodedata.normalize("NFKD", text)  # Normalize unicode
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
#     text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     return text

# # ✅ Extract text from PDF files
# def extract_text_from_pdf(file):
#     doc = pymupdf.open(stream=file.read(), filetype="pdf")
#     return "\n".join([page.get_text("text") for page in doc])

# # ✅ Extract text from Word (DOCX) files
# def extract_text_from_docx(file):
#     doc = Document(file)
#     return "\n".join([para.text for para in doc.paragraphs])

# # ✅ Load the trained model & TF-IDF vectorizer
# if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
#     model = joblib.load(MODEL_PATH)
#     tfidf_vectorizer = joblib.load(VECTORIZER_PATH)  # Load the vectorizer
# else:
#     st.error("Model or vectorizer file not found! Ensure 'logistic_randomforest.pkl' and 'tfidf_vectorizer.pkl' exist.")
#     st.stop()

# # ✅ Prediction function (Fixed ValueError)
# def predict_job_real_or_fake(description):
#     preprocessed_text = preprocess_text(description)  # Clean text
#     transformed_text = tfidf_vectorizer.transform([preprocessed_text])  # Convert to TF-IDF
#     return model.predict(transformed_text)[0]  # Predict using trained model

# # ✅ Streamlit UI
# def main():
#     st.set_page_config(page_title="Job Authenticity Checker", layout="wide")
#     st.markdown(
#         """
#         <style>
#             .main-title {
#                 font-size: 32px;
#                 font-weight: bold;
#                 text-align: center;
#                 color: #FFD700;
#             }
#             .sidebar .sidebar-content {
#                 background: linear-gradient(to bottom, #1e1e2e, #282a36);
#             }
#             .stButton>button {
#                 background: #4CAF50;
#                 color: white;
#                 font-size: 18px;
#                 border-radius: 8px;
#                 padding: 10px;
#                 width: 100%;
#             }
#             .stTextInput, .stTextArea {
#                 border-radius: 8px;
#                 border: 1px solid #ccc;
#             }
#             .prediction-box {
#                 text-align: center;
#                 font-size: 20px;
#                 font-weight: bold;
#                 padding: 15px;
#                 border-radius: 8px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     st.markdown('<p class="main-title">🔍 Job Posting Authenticity Checker</p>', unsafe_allow_html=True)
#     st.sidebar.title("Navigation")
    
#     menu = ["Home", "Stored Jobs", "About"]
#     choice = st.sidebar.radio("Go to", menu)
    
#     if choice == "Home":
#         st.subheader("Upload Job Details")
#         uploaded_file = st.file_uploader("Upload job description (PDF/DOCX)", type=["pdf", "docx"], key="file_uploader")
#         job_title, description = "", ""
        
#         if uploaded_file:
#             file_type = uploaded_file.type
#             if "pdf" in file_type:
#                 file_content = extract_text_from_pdf(uploaded_file)
#             elif "word" in file_type or "msword" in file_type:
#                 file_content = extract_text_from_docx(uploaded_file)

#             file_lines = file_content.split("\n", 1)
#             if len(file_lines) >= 2:
#                 job_title, description = file_lines[0], file_lines[1]
#             else:
#                 st.warning("Invalid file format. Ensure the first line is the job title.")
#                 return
#         else:
#             job_title = st.text_input("Job Title")
#             description = st.text_area("Job Description", height=200)
        
#         if st.button("Predict", use_container_width=True):
#             if job_title and description:
#                 prediction = predict_job_real_or_fake(description)
#                 result_text = "✅ Job Posting is Real" if prediction == 0 else "❌ Likely Fake"
#                 color = "#4CAF50" if prediction == 0 else "#FF5733"
#                 st.markdown(f'<p class="prediction-box" style="background-color:{color}; color:white;">{result_text}</p>', unsafe_allow_html=True)
#                 store_to_db(job_title, description, prediction)
#             else:
#                 st.warning("Please enter a job title and description.")
    
#     elif choice == "Stored Jobs":
#         st.subheader("Stored Job Postings")
#         df = fetch_stored_jobs()
#         if not df.empty:
#             st.dataframe(df, use_container_width=True)
#         else:
#             st.warning("No job postings stored yet.")
    
#     elif choice == "About":
#         st.subheader("About This App")
#         st.info("This app helps in detecting fake job postings using machine learning models.")

# if __name__ == "__main__":
#     initialize_db()
#     main()



# import re
# import string
# import unicodedata
# import pymupdf  # PyMuPDF for PDF processing
# from docx import Document  # python-docx for Word processing

# # st.set_page_config(page_title="Job Authenticity Checker", layout="wide")

# # Set base directimport os
# import joblib
# import streamlit as st
# import pandas as pd
# import sqlite3ory for model and vectorizer
# BASE_DIR = r"C:\xampp_inuse\htdocs\web\formvalidation\prediction"
# MODEL_PATH = os.path.join(BASE_DIR, "logistic_newdataset (2).pkl")
# VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer_newdataset.pkl")
# DB_PATH = os.path.join(BASE_DIR, "job_postings.db")

# # Initialize database
# def initialize_db():
#     with sqlite3.connect(DB_PATH) as conn:
#         conn.execute('''CREATE TABLE IF NOT EXISTS postings 
#                         (id INTEGER PRIMARY KEY, 
#                         job_title TEXT, 
#                         description TEXT, 
#                         prediction INTEGER)''')

# # Load model and vectorizer
# if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
#     model = joblib.load(MODEL_PATH)
#     tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
# else:
#     st.error(f"🚨 Model or vectorizer file not found!\nExpected at:\n {MODEL_PATH}\n {VECTORIZER_PATH}")
#     st.stop()

# # ✅ Function to preprocess job descriptions
# def preprocess_text(text):
#     text = text.lower()
#     text = unicodedata.normalize("NFKD", text)
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
#     text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     return text

# # ✅ Function to extract text from uploaded PDF/DOCX
# def extract_text_from_file(uploaded_file):
#     text = ""
#     if uploaded_file is not None:
#         file_extension = uploaded_file.name.split(".")[-1]
#         if file_extension == "pdf":
#             pdf_doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
#             text = " ".join([page.get_text() for page in pdf_doc])
#         elif file_extension == "docx":
#             doc = Document(uploaded_file)
#             text = " ".join([para.text for para in doc.paragraphs])
#     return text.strip()

# def main():
#     st.markdown('<p class="main-title">🔍 Job Posting Authenticity Checker</p>', unsafe_allow_html=True)
    
#     menu = ["Home", "Stored Jobs", "About"]
#     choice = st.sidebar.radio("Go to", menu)
    
#     if choice == "Home":
#         st.subheader("Upload or Enter Job Details")
        
#         uploaded_file = st.file_uploader("Upload job description (PDF/DOCX)", type=["pdf", "docx"])
#         job_title = st.text_input("Job Title (Optional)")
#         manual_description = st.text_area("Or Enter Job Description Manually", height=200)

#         # ✅ Extract text from uploaded file if provided
#         extracted_text = extract_text_from_file(uploaded_file) if uploaded_file else ""

#         # ✅ Prioritize uploaded text, otherwise use manual input
#         final_description = extracted_text if extracted_text else manual_description

#         if st.button("Predict", use_container_width=True):
#             if final_description:  # ✅ Only require job description, job title is optional
#                 processed_text = preprocess_text(final_description)
#                 prediction = model.predict(tfidf_vectorizer.transform([processed_text]))[0]
#                 result_text = "✅ Job Posting is Real" if prediction == 0 else "❌ Likely Fake"
#                 st.markdown(f'<p style="background-color:#4CAF50; padding:10px; border-radius:8px;">{result_text}</p>', unsafe_allow_html=True)
#             else:
#                 st.warning("⚠️ Please upload a file OR enter a job description.")

# if __name__ == "__main__":
#     initialize_db()
#     main()

import streamlit as st
import time
import random

# Set page configuration
st.set_page_config(page_title="Job Authenticity Checker", layout="wide")

# Custom CSS for a modern, colorful, and professional design with enhanced animations
st.markdown("""
    <style>
        /* App-wide styling */
        .stApp {
            background: linear-gradient(135deg, #FFE1E1 0%, #E1F5FF 100%); /* Gradient from soft pink to light blue */
            font-family: 'Roboto', sans-serif;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        /* Title and subtitle */
        h1 {
            color: #D81B60; /* Vibrant Pink for title */
            display: flex;
            align-items: center;
            gap: 10px;
            animation: fadeIn 1s ease-in-out;
        }
        p {
            text-align: center;
        }

        /* Animations */
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }

        @keyframes slideInLeft {
            0% { transform: translateX(-30px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }

        @keyframes slideInRight {
            0% { transform: translateX(30px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        @keyframes fadeText {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }

        @keyframes bounce {
            0% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0); }
        }

        @keyframes waveBar {
            0% { transform: scaleY(1); background-color: #FF6F61; }
            25% { transform: scaleY(2); background-color: #FFCA28; }
            50% { transform: scaleY(1); background-color: #AB47BC; }
            75% { transform: scaleY(1.5); background-color: #0288D1; }
            100% { transform: scaleY(1); background-color: #FF6F61; }
        }

        @keyframes loadingGradient {
            0% { background: rgba(40, 53, 147, 0.9); }
            50% { background: rgba(74, 20, 140, 0.9); }
            100% { background: rgba(40, 53, 147, 0.9); }
        }

        /* Card Styling for Sections */
        .card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            transition: transform 0.3s ease;
            border: 2px solid #FFAB91; /* Soft coral border for cards */
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
            border-color: #FF6F61; /* Darker coral on hover */
        }
        .card-left {
            animation: slideInLeft 0.8s ease-out;
        }
        .card-right {
            animation: slideInRight 0.8s ease-out;
        }

        /* Section Headers */
        h3 {
            color: #0288D1; /* Bright blue for section headers */
            font-weight: 700;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 18px;
        }

        /* Input Fields */
        .stTextInput > div > input,
        .stTextArea > div > textarea,
        .stFileUploader {
            border-radius: 8px;
            background-color: #F3E5F5 !important; /* Light purple background */
            border: 1px solid #AB47BC !important; /* Purple border */
            color: #4A148C !important; /* Dark purple text */
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .stTextInput > div > input:focus,
        .stTextArea > div > textarea:focus {
            border-color: #7B1FA2 !important; /* Darker purple on focus */
            box-shadow: 0 0 5px rgba(123, 31, 162, 0.3);
        }
        .stFileUploader label {
            color: #6B7280;
        }

        /* Button Styling with Pulse Animation */
        .stButton > button {
            background-color: #FF6F61 !important; /* Coral button */
            color: white !important;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton > button:hover {
            background-color: #FF3D00 !important; /* Darker coral on hover */
            animation: pulse 0.5s;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        }

        /* Centered Loading Box with Wave Animation */
        .loading-box {
            width: 100%;
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            background: rgba(40, 53, 147, 0.9); /* Indigo overlay with gradient */
            animation: loadingGradient 6s infinite;
            color: #FFFFFF;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 24px;
            font-weight: 600;
            z-index: 9999;
            flex-direction: column;
        }
        .wave-container {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }
        .wave-bar {
            width: 10px;
            height: 30px;
            background-color: #FF6F61; /* Starting color */
            border-radius: 5px;
            animation: waveBar 2s infinite;
        }
        .wave-bar:nth-child(2) {
            animation-delay: 0.2s;
        }
        .wave-bar:nth-child(3) {
            animation-delay: 0.4s;
        }
        .wave-bar:nth-child(4) {
            animation-delay: 0.6s;
        }
        .wave-bar:nth-child(5) {
            animation-delay: 0.8s;
        }
        .ball {
            width: 10px;
            height: 10px;
            background-color: #FFCA28; /* Yellow balls */
            border-radius: 50%;
            margin: 5px;
            animation: bounce 1.5s infinite;
            position: absolute;
        }
        .ball:nth-child(2) {
            animation-delay: 0.5s;
        }
        .ball:nth-child(3) {
            animation-delay: 1s;
        }
        .fade-text {
            animation: fadeText 2s infinite;
        }

        /* Highlight uploaded file */
        .highlight-file {
            background-color: #FFECB3; /* Light yellow */
            padding: 10px 15px;
            border-radius: 8px;
            font-weight: 600;
            color: #E65100; /* Orange text */
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        /* Result Animation (Slide Up) */
        .result-container {
            animation: slideUp 0.5s ease-out;
        }

        /* Success and Error Messages */
        .stSuccess, .stError {
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            font-weight: 500;
        }
        .stSuccess {
            background-color: #C8E6C9; /* Light green */
            border: 1px solid #4CAF50; /* Green border */
            color: #1B5E20; /* Dark green text */
        }
        .stError {
            background-color: #FFCDD2; /* Light red */
            border: 1px solid #EF5350; /* Red border */
            color: #B71C1C; /* Dark red text */
        }

        /* Column spacing */
        .stColumns > div {
            padding: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)

# Main application logic
with st.container():
    # Title
    st.title("🔍 Job Authenticity Checker")

    # Two-column layout for better organization
    col1, col2 = st.columns([1, 1], gap="medium")

    # File upload section
    with col1:
        with st.container():
            st.markdown('<div class="card card-left">', unsafe_allow_html=True)
            st.markdown("### Upload Job Description", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Supports PDF/DOCX files", type=["pdf", "docx"])
            if uploaded_file:
                st.markdown(f'<div class="highlight-file">📄 {uploaded_file.name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Manual input section
    with col2:
        with st.container():
            st.markdown('<div class="card card-right">', unsafe_allow_html=True)
            st.markdown("### Or Enter Manually", unsafe_allow_html=True)
            job_title = st.text_input("Job Title (Optional)", placeholder="Enter job title...")
            job_description = st.text_area("Job Description", placeholder="Paste job description here...", height=150)
            st.markdown('</div>', unsafe_allow_html=True)

    # Center the button below columns
    st.markdown("<div style='display: flex; justify-content: center; margin-top: 20px;'>", unsafe_allow_html=True)
    if st.button("Analyze Now"):
        if not uploaded_file and not job_description.strip():
            st.error("⚠️ Please upload a file or enter a job description before submitting.")
        else:
            # Display loading message with wave animation
            loading_placeholder = st.empty()
            loading_placeholder.markdown(
                '<div class="loading-box"><div class="wave-container"><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div></div><span class="fade-text">Processing Your Job Check! 🎉 Stay Tuned... 🚀</span><div class="ball"></div><div class="ball"></div><div class="ball"></div></div>',
                unsafe_allow_html=True
            )

            # Simulate processing time (8-10 seconds)
            time.sleep(random.randint(8, 10))

            # Remove loading message
            loading_placeholder.empty()

            # Dummy Result
            result = random.choice([True, False])

            # Display Result with animation
            result_placeholder = st.empty()
            with result_placeholder.container():
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                if result:
                    st.success("✅ The job posting seems legitimate!")
                    st.markdown("**Next Steps:** Proceed with confidence, but always double-check details.")
                else:
                    st.error("⚠️ This job posting may be fake! 🚨")
                    st.markdown("**Recommendation:** Verify the source and contact the employer directly.")
                st.markdown('</div>', unsafe_allow_html=True)

            # Highlight the uploaded file name again (if present)
            if uploaded_file:
                st.markdown(f'<div class="highlight-file">📄 {uploaded_file.name}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)