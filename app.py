import streamlit as st
from PyPDF2 import PdfReader
# import langchain.text_splitter as RecursiveCharacterTextSplitter
# Prefer split-out package, fallback to legacy path
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Optional: use LangChain's Google chat model for compatibility with load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Page config and minimal CSS for better UX
st.set_page_config(page_title="PDF Question Answering", page_icon="📚", layout="wide")
st.markdown("""
<style>
    .app-title { font-size: 2rem; font-weight: 700; margin-bottom: .25rem; }
    .app-sub { color: #666; margin-bottom: 1rem; }
    .panel { background: #f7f9fc; padding: 1rem 1.25rem; border-radius: 10px; border: 1px solid #e9eef5; }
    .response { background: #fff; border-left: 4px solid #1f77b4; padding: .75rem 1rem; border-radius: 6px; }
    .small { font-size: .9rem; color: #777; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Ensure env vars are loaded before API configuration
load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Missing GEMINI_API_KEY in .env (or set GOOGLE_API_KEY).")
    st.stop()
# Ensure the LangChain wrappers see the key (they read GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

# Session state flags for UX
if "vs_ready" not in st.session_state:
    st.session_state.vs_ready = False
if "last_response" not in st.session_state:
    st.session_state.last_response = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n"
    return text

def get_text_chunks(text):
    # USE: RecursiveCharacterTextSplitter from fixed import
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("vector_store")


def get_conversation_chain():
    prompt_template = "Answer the question based on the context provided. If the context does not contain the answer, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}"
    # USE: LangChain chat model for QA chain compatibility
    model = ChatGoogleGenerativeAI(
        model=os.environ.get("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.3,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    # Add safe flag for FAISS (required in newer versions)
    new_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    # Return the text so the UI can render it nicely
    return response['output_text']

def main():
    # Header
    st.markdown('<div class="app-title">📚 PDF Question Answering</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Upload PDFs, build a knowledge base, and ask questions.</div>', unsafe_allow_html=True)

    # Sidebar workflow
    with st.sidebar:
        st.header("1) Upload & Process")
        pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        process = st.button("Process documents", use_container_width=True, disabled=not pdf_docs)
        clear_vs = st.button("Clear vector store", use_container_width=True)

        if clear_vs:
            st.session_state.vs_ready = False
            st.session_state.last_response = None
            import shutil
            shutil.rmtree("vector_store", ignore_errors=True)
            st.success("Cleared local vector store.")

        if process and pdf_docs:
            with st.spinner("Reading and indexing PDFs..."):
                try:
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    get_vector_store(chunks)
                    st.session_state.vs_ready = True
                    st.success("Vector store ready.")
                except Exception as e:
                    st.session_state.vs_ready = False
                    st.error(f"Failed to process PDFs: {e}")

        st.markdown("---")
        st.header("2) Ask")
        st.caption("Once processed, ask questions about the content.")

    # Main interaction
    col_q, col_info = st.columns([3, 2])
    with col_q:
        q = st.text_input(
            "Ask a question",
            placeholder="e.g., What are the main conclusions in the documents?",
            disabled=not st.session_state.vs_ready
        )
        ask = st.button("Ask", type="primary", disabled=not st.session_state.vs_ready)

        if not st.session_state.vs_ready:
            st.info("Upload and process PDFs in the sidebar to enable questions.")

        if ask and q:
            with st.spinner("Thinking..."):
                try:
                    answer = user_input(q)
                    st.session_state.last_response = answer
                except Exception as e:
                    st.error(f"Error while answering: {e}")
                    st.session_state.last_response = None

        if st.session_state.last_response:
            st.markdown("**Answer**")
            st.markdown(f'<div class="response">{st.session_state.last_response}</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Status**")
        st.write("Vector store:", "Ready ✅" if st.session_state.vs_ready else "Not ready ❌")
        st.markdown('<span class="small">Tip: You can clear and reprocess documents from the sidebar.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Run
if __name__ == "__main__":
    main()