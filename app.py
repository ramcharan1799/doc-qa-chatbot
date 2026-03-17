import streamlit as st
from utils.loader import load_pdf, chunk_text
from utils.qa_engine import build_index, retrieve, answer

st.set_page_config(page_title="Doc Q&A", page_icon="📄", layout="centered")

st.title("📄 Document Q&A Chatbot")
st.caption("Upload a PDF and ask questions about it.")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file:
        if "index" not in st.session_state or \
           st.session_state.get("filename") != uploaded_file.name:
            with st.spinner("Reading and indexing document..."):
                pages = load_pdf(uploaded_file)
                chunks = chunk_text(pages)
                index, chunks = build_index(chunks)
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.filename = uploaded_file.name
                st.session_state.messages = []
            st.success(f"Indexed {len(chunks)} chunks from {len(pages)} pages")
    st.divider()
    st.caption("Built with OpenAI + FAISS + Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if "index" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = retrieve(prompt, st.session_state.index, st.session_state.chunks)
                reply = answer(prompt, context)
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
