# transformers_lib_chatbot_using_googles_bert_&_t5_llms.py

import os
import PyPDF2
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BertTokenizer,
    BertForQuestionAnswering,
    pipeline
)

# Initialize models and tokenizers
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
qa_pipeline = pipeline('question-answering', model='bert-large-uncased', tokenizer='bert-large-uncased')


def extract_text_from_pdf_fitz(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def extract_text_from_pdf_pypdf2(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def summarize_text(text):
    """Summarize the given text using T5."""
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = t5_model.generate(
        inputs,
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def answer_question(question, context):
    """Answer a question based on the context using BERT."""
    result = qa_pipeline(question=question, context=context)
    return result['answer']


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PDF QA + Summary using BERT and T5")
    parser.add_argument('--pdf', required=True, help='Path to the PDF file')
    parser.add_argument('--extractor', choices=['fitz', 'pypdf2'], default='pypdf2', help='Choose the PDF text extraction method')
    parser.add_argument('--question', help='Ask a question based on the PDF content')
    parser.add_argument('--summary', action='store_true', help='Summarize the content of the PDF')

    args = parser.parse_args()

    # Extract PDF text
    if args.extractor == 'fitz':
        text = extract_text_from_pdf_fitz(args.pdf)
    else:
        text = extract_text_from_pdf_pypdf2(args.pdf)

    if args.summary:
        print("\n=== Summary ===")
        print(summarize_text(text))

    if args.question:
        print("\n=== Question Answering ===")
        print(f"Q: {args.question}")
        print(f"A: {answer_question(args.question, text)}")


if __name__ == "__main__":
    main()
