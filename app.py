question = st.text_input("Ask a question about your document:")

if question:
    question_embedding = model.encode([question])

    D, I = index.search(
        np.array(question_embedding).astype("float32"),
        k=3
    )

    answer = ""

    for idx in I[0]:
        answer += chunks[idx] + "\n"

    st.subheader("Answer")
    st.write(answer)