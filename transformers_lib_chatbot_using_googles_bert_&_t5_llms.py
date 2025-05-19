# Install necessary libraries
!pip install streamlit PyPDF2 transformers

# pdf_chatbot_streamlit.py

import streamlit as st
import PyPDF2
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# Initialize models and tokenizers
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
qa_pipeline = pipeline('question-answering', model='bert-large-uncased', tokenizer='bert-large-uncased')

def extract_text_from_pdf(pdf_file):
    """Extract text from the uploaded PDF file."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def summarize_text(text):
    """Summarize the given text using T5."""
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = t5_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def answer_question(question, context):
    """Answer a question based on the context using BERT."""
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit app
st.set_page_config(page_title="PDF Chatbot with Transformers", layout="wide")
st.title("PDF Summarizer and Q&A using BERT & T5")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)

    if text:
        st.subheader("Extracted Text (First 500 Characters)")
        display_text = text[:500] + ('...' if len(text) > 500 else '')
        st.text_area("Extracted Text", value=display_text, height=250)

        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(text)
                st.subheader("Summary")
                st.write(summary)

        question = st.text_input("Ask a question based on the PDF content")
        if question:
            with st.spinner("Finding answer..."):
                answer = answer_question(question, text)
                st.subheader("Answer")
                st.write(answer)
else:
    st.info("Please upload a PDF file to begin.")
