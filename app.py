# app.py

import streamlit as st
from utils import extract_text_from_pdf, build_rag_chain
import os

st.set_page_config(page_title="🧑‍💼 HR Policy Chatbot", layout="centered")
st.title("🤖 HR Policy Chatbot (RAG)")

st.markdown("Upload your HR policy PDF or use the sample one provided.")

# Option 1: Upload a PDF
uploaded_file = st.file_uploader("📁 Upload a HR PDF", type=["pdf"])

# Option 2: Load default PDF from folder
use_sample = st.checkbox("Use sample TCS HR PDF instead")

if uploaded_file or use_sample:
    with st.spinner("Reading and indexing PDF..."):
        if uploaded_file:
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            default_pdf_path = os.path.join("sample_docs", "TCS_HR_Digital_Transformation.pdf")
            with open(default_pdf_path, "rb") as f:
                raw_text = extract_text_from_pdf(f)

        rag_chain = build_rag_chain(raw_text)
        st.success("✅ PDF processed! Ask your HR question below.")

    question = st.text_input("❓ Ask a question based on HR policy")

    if question:
        with st.spinner("Searching for answer..."):
            response = rag_chain.run(question)
            st.success("Answer:")
            st.write(response)
