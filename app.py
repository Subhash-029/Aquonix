import streamlit as st
import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import torch

# ✅ Load Embedding Model for SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

st.title("Ask Aquonix")

# Custom CSS for chat messages
st.markdown("""
    <style>
        .user-message {
            text-align: right;
            background-color: #3c8ce7;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: inline-block;
            width: fit-content;
            max-width: 70%;
            margin-left: auto;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .assistant-message {
            text-align: left;
            background-color: #d16ba5;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: inline-block;
            width: fit-content;
            max-width: 70%;
            margin-right: auto;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

# Sidebar for model selection and user details
st.sidebar.write("Settings")
st.sidebar.write("-----------")
model_options = ["MBZUAI/LaMini-T5-738M", "google/flan-t5-base", "google/flan-t5-small"]
selected_model = st.sidebar.radio("Choose Model", model_options)
st.sidebar.write("-----------")
uploaded_file = st.sidebar.file_uploader("Upload file", type=["pdf"])
st.sidebar.write("-----------")
st.sidebar.write("About Me")
st.sidebar.write("Name: Deepak Yadav")
st.sidebar.write("Bio: Passionate about AI and machine learning. Enjoys working on innovative projects and sharing knowledge with the community.")
st.sidebar.write("[GitHub](https://github.com/Subhash-029)")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/dky7376/)")
st.sidebar.write("-----------")

@st.cache_resource
def initialize_qa_chain(filepath, model_name):
    try:
        loader = PDFMinerLoader(filepath)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # ✅ Create embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(splits, embeddings)

        # ✅ Load model with proper error handling
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            device_map="auto",  # ✅ Auto-select device for efficiency
            torch_dtype=torch.float32  # ✅ Reduce memory usage
        )

        text_gen_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
        )

        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

        # ✅ Build QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing QA Chain: {str(e)}")
        return None

def process_answer(instruction, qa_chain):
    if qa_chain:
        return qa_chain.run(instruction)
    return "Error: QA Chain is not initialized."

if uploaded_file is not None:
    os.makedirs("docs", exist_ok=True)
    filepath = os.path.join("docs", uploaded_file.name)
    with open(filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filepath = temp_file.name

    with st.spinner('Processing document...'):
        qa_chain = initialize_qa_chain(temp_filepath, selected_model)
else:
    qa_chain = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("What is up?"):
    st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if qa_chain:
        response = process_answer({'query': prompt}, qa_chain)
    else:
        response = "Please upload a PDF file to enable the chatbot."

    st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})
