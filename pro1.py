

import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("API")

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_api_key  # Put your key in an environment variable
)


0

def create_pdf(text):
    pdf_file = "Improved_Resume.pdf"

    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()

    story = []

    for line in text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)

    return pdf_file


# ---------------- Resume Analyzer ---------------- #

def getans(uploaded_pdf):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)

    docs = loader.load()

    resume = "\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert ATS Resume Reviewer.

Analyze the resume.

Return the output in EXACTLY this format.

# ATS_SCORE

# STRENGTHS

# WEAKNESSES

# IMPROVEMENTS

# IMPROVED_RESUME

(Write ONLY the improved and marked down resume here)

# INTERVIEW_QUESTIONS

Resume:

{resume}
"""
    )

    chain = prompt | llm

    return chain.invoke({"resume": resume})


# ---------------- Streamlit ---------------- #

st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

if uploaded_file is not None:

    with st.spinner("Analyzing Resume..."):
        ans = getans(uploaded_file)

    st.markdown(ans.content)

    # -------- Extract Improved Resume -------- #

    text = ans.content

    start = text.find("# IMPROVED_RESUME")
    end = text.find("# INTERVIEW_QUESTIONS")

    if start != -1 and end != -1:

        improved_resume = text[
            start + len("# IMPROVED_RESUME"):end
        ].strip()

        pdf_path = create_pdf(improved_resume)

        with open(pdf_path, "rb") as f:

            st.download_button(
                label="📥 Download Improved Resume",
                data=f,
                file_name="Improved_Resume.pdf",
                mime="application/pdf"
            )