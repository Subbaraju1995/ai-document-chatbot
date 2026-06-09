import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

st.title("AI Document Chatbot")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

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

        context = ""
        for idx in I[0]:
            context += chunks[idx] + "\n\n"

        prompt = f"""
        You are an AI document assistant.
        Answer the user's question using only the context below.
        If the answer is not in the document, say: "I could not find that information in the document."

        Context:
        {context}

        Question:
        {question}

        Answer clearly in simple language:
        """

        response = gemini_model.generate_content(prompt)

        st.subheader("AI Answer")
        st.write(response.text)

        with st.expander("View retrieved document context"):
            st.write(context)