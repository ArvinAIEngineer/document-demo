import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import pickle
from dotenv import load_dotenv
import warnings
import base64

# Disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

# Load environment variables
load_dotenv()

def display_pdf(pdf_path):
    """Display the PDF in the Streamlit app using base64 encoding."""
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("The PDF file was not found in the root folder.")
    except Exception as e:
        st.error(f"Error displaying the PDF: {e}")

@st.cache_resource
def initialize_groq_llm():
    """Initialize Groq LLM with API key"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("API key is missing. Please check your .env file.")
        return None
    return Groq(api_key=api_key)

def load_pdf_text(file_path):
    """Extract text from PDF"""
    try:
        pdf_reader = PdfReader(file_path)
        return "".join([page.extract_text() or "" for page in pdf_reader.pages])
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def load_vector_store(pdf_text, embeddings, store_name):
    """Load or create vector store"""
    pickle_file = f"{store_name}.pkl"
    
    if os.path.exists(pickle_file):
        try:
            with open(pickle_file, "rb") as f:
                vector_store = pickle.load(f)
                if not isinstance(vector_store, FAISS):
                    raise TypeError("Invalid vector store format")
                return vector_store
        except Exception as e:
            st.error(f"Vector store loading error: {e}")
            if os.path.exists(pickle_file):
                os.remove(pickle_file)
            return None
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=pdf_text)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        with open(pickle_file, "wb") as f:
            pickle.dump(vector_store, f)
        return vector_store
    except Exception as e:
        st.error(f"Vector store creation error: {e}")
        return None

def get_answer(query, vector_store, llm):
    """Get answer for a query"""
    try:
        docs = vector_store.similarity_search(query=query, k=3)
        snippets = " ".join([doc.page_content for doc in docs])
        
        prompt = f"Given the following document snippets, Provide short and crisp one line response to the query: '{query}'.\n\nDocument Snippets:\n{snippets}"
        
        result = llm.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Provide short and crisp one line answers based on the provided document snippets."},
                {"role": "user", "content": prompt}
            ]
        )
        return result.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def main():
    st.title("📄 Vikas group Demo")
    
    # Initialize LLM
    llm = initialize_groq_llm()
    if llm is None:
        return

    # PDF Path and Display
    pdf_path = "document.pdf"  # Adjust to your PDF filename

    # Show PDF preview
    st.subheader("Document Preview")
    try:
        display_pdf(pdf_path)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")
        return

    # Load PDF text
    pdf_text = load_pdf_text(pdf_path)
    if not pdf_text:
        return

    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = load_vector_store(pdf_text, embeddings, "vector_store")
    if not vector_store:
        return

    # Preset questions
    st.subheader("Common Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What is the cooling capacity?"):
            with st.spinner("Analyzing..."):
                response = get_answer("What is the cooling capacity required?", vector_store, llm)
                st.info(response)
    
    with col2:
        if st.button("How much is the energy efficiency ratio?"):
            with st.spinner("Analyzing..."):
                response = get_answer("How much is the energy efficiency ratio?", vector_store, llm)
                st.info(response)
    
    with col3:
        if st.button("What is the delay penalty?"):
            with st.spinner("Analyzing..."):
                response = get_answer("How much is the delivery delay penalty?", vector_store, llm)
                st.info(response)

    # Custom question input
    st.subheader("Ask Your Own Question")
    custom_question = st.text_input("Enter your question about the document:")
    if custom_question:
        with st.spinner("Analyzing..."):
            response = get_answer(custom_question, vector_store, llm)
            st.info(response)

if __name__ == '__main__':
    main()
