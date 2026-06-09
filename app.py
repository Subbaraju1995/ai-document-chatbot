import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("AI Document Chatbot")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)

    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("PDF uploaded successfully!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)
    st.write(f"Total Chunks Created: {len(chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    st.success("Embeddings created and stored in FAISS!")

    question = st.text_input("Ask a question about your document:")

    if question:
        question_embedding = model.encode([question]).astype("float32")

        D, I = index.search(question_embedding, k=3)

        answer = ""
        for idx in I[0]:
            answer += chunks[idx] + "\n\n"

        st.subheader("Answer")
        st.write(answer)