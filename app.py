import streamlit as st
from streamlit_lottie import st_lottie
import requests
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pinecone import Pinecone, PodSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
import json

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text

def get_text_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunks = text_splitter.split_text(text)
  return chunks

def get_or_create_index(index_name, dimension):
    try:
        # Check if the index already exists
        if index_name in pc.list_indexes().names():
            print(f"Using existing index: {index_name}")
            return pc.Index(index_name)
        
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=PodSpec(environment=os.getenv("PINECONE_ENVIRONMENT"))
        )
        return pc.Index(index_name)
    except Exception as e:
        print(f"Error with index: {e}")
        existing_indexes = pc.list_indexes().names()
        if existing_indexes:
            print(f"Using existing index: {existing_indexes[0]}")
            return pc.Index(existing_indexes[0])
        else:
            raise Exception("No indexes available and unable to create a new one.")

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    index_name = "rohitdb"  # Choose a consistent name for your index
    dimension = 768  # Adjust this based on your embedding model
    
    index = get_or_create_index(index_name, dimension)
    
    # Create the vector store
    vector_store = LangchainPinecone.from_existing_index(index_name, embeddings)
    
    # Upsert the text chunks
    vector_store.add_texts(text_chunks)
    
    return vector_store

def get_conversation_chain(vector_store):
    prompt_template = """
    You are a helpful assistant that can answer questions about the provided documents. Make sure to provide all the details.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    Context: {context}
    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        top_p=0.85,
        top_k=40
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    index_name = "rohitdb"  # Use the same name as above
    vector_store = LangchainPinecone.from_existing_index(index_name, embeddings)
    
    qa_chain = get_conversation_chain(vector_store)
    
    response = qa_chain({"query": user_question})
    
    return response["result"]

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_lottie_file(url, filepath):
    r = requests.get(url)
    if r.status_code == 200:
        with open(filepath, 'w') as f:
            json.dump(r.json(), f)
        return True
    return False

def load_lottie_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)

def setup_lottie_files():
    lottie_dir = "lottie_files"
    ensure_dir(lottie_dir)
    
    lottie_urls = {
        "file_upload": "https://assets9.lottiefiles.com/packages/lf20_iorpbol0.json",
        "ai_chat": "https://assets3.lottiefiles.com/packages/lf20_26KVvs.json",
        "success": "https://assets9.lottiefiles.com/packages/lf20_lk80fpsm.json",
        "book": "https://assets5.lottiefiles.com/packages/lf20_1a8dx7zj.json"
    }
    
    for name, url in lottie_urls.items():
        filepath = os.path.join(lottie_dir, f"{name}.json")
        if not os.path.exists(filepath):
            if download_lottie_file(url, filepath):
                print(f"Downloaded {name} Lottie file")
            else:
                print(f"Failed to download {name} Lottie file")

def main():
    setup_lottie_files()
    
    st.set_page_config(page_title="Chat PDF", page_icon="📚", layout="wide")

    # Custom CSS for animations and styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    /* Styling for the input field */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #333333;
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px 15px;
        font-size: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #45a049;
        box-shadow: 0 0 0 2px rgba(76,175,80,0.2);
        outline: none;
    }
    /* Styling for the Ask button */
    .ask-button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    .ask-button:hover {
        background-color: #45a049 !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📁 Document Upload")
        
        # Add a Lottie animation for file upload
        lottie_upload = load_lottie_file("lottie_files/file_upload.json")
        if lottie_upload:
            st_lottie(lottie_upload, speed=1, height=200, key="upload_animation")
        
        pdf_docs = st.file_uploader(
            "Upload your PDF file",
            accept_multiple_files=True,
            help="Select one PDF files to upload"
        )
        
        if st.button("Process Documents", key="process_btn"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    # Simulate processing time with a progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    
                st.success("Documents processed successfully!")
                st.balloons()
            else:
                st.warning("Please upload at least one PDF file.")

    # Main content
    st.title("Chat with PDF using Gemini 💬")
    
    # Add a Lottie animation for AI chat
    lottie_chat = load_lottie_file("lottie_files/ai_chat.json")
    if lottie_chat:
        st_lottie(lottie_chat, speed=1, height=200, key="chat_animation")
    
    st.markdown("---")

    # Initialize session state for user question and answer if they don't exist
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ''
    if 'answer' not in st.session_state:
        st.session_state.answer = ''

    # Create two columns for the input field and button
    col1, col2 = st.columns([3, 1])

    with col1:
        user_question = st.text_input("Ask a question about the uploaded PDF:", key="user_question_input", value=st.session_state.user_question)

    with col2:
        ask_button = st.button("Ask", key="ask_button", help="Click to submit your question", use_container_width=True)

    if ask_button and user_question:
        st.session_state.user_question = user_question
        with st.spinner("Thinking..."):
            answer = user_input(user_question)
            st.session_state.answer = answer

    if st.session_state.answer:
        st.markdown("### Answer:")
        st.write(st.session_state.answer)
        
        # Add a Lottie animation for successful answer
        lottie_success = load_lottie_file("lottie_files/success.json")
        if lottie_success:
            st_lottie(lottie_success, speed=1, height=200, key="success_animation")

    # Add a Lottie animation at the bottom for decoration
    lottie_book = load_lottie_file("lottie_files/book.json")
    if lottie_book:
        st_lottie(lottie_book, speed=0.5, height=300, key="book")

if __name__ == "__main__":
    main()
else:
    # Create a new Streamlit app
    app = st.create_app()
    app.add_app("main", main)
