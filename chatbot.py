# Libraries for file handling, logging, environment variable loading
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import requests
# Libraries for web application and HTTP requests
import streamlit as st
from streamlit_chat import message
# Libraries for document loading, embeddings, AI agents and predictions
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.agents import create_csv_agent
from langchain.embeddings import HuggingFaceEmbeddings
from OctoAiCloudLLM import OctoAiCloudLLM
from llama_index import (LLMPredictor, ServiceContext,
                         download_loader, GPTVectorStoreIndex, LangchainEmbedding)

# Load environment variables and set page configurations
load_dotenv()
st.set_page_config(
    page_title="OctoAI Movie Bot - Demo",
    page_icon=":robot:"
)

# Set global variables
os.environ["OCTOAI_API_TOKEN"] = st.secrets['OCTOAI_API_TOKEN']
os.environ["ENDPOINT_URL"] = st.secrets['ENDPOINT_URL']
endpoint_url = os.getenv("ENDPOINT_URL")

st.header("Movie Bot - Demo")

# Function to handle session state


def handle_session_state():
    # Use session_state to manage data across reruns
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'count' not in st.session_state:
        st.session_state.count = 0
        st.session_state.chat_history_ids = None
        st.session_state.old_response = ''


handle_session_state()

# Load data from csv file
file = Path('rotten_tomatoes_top_movies.csv')
PagedCSVReader = download_loader("PagedCSVReader")
loader = PagedCSVReader()
documents = loader.load_data(file)

# Initialize the OctoAiCloudLLM and LLMPredictor
llm = OctoAiCloudLLM(endpoint_url=endpoint_url)
llm_predictor = LLMPredictor(llm=llm)

# Create the LangchainEmbedding
if 'embeddings' not in st.session_state:
    embeddings = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    st.session_state.embeddings = embeddings
else:
    embeddings = st.session_state.embeddings
# Create the ServiceContext
if 'service_context' not in st.session_state:
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=1024, embed_model=embeddings)
    st.session_state.service_context = service_context
else:
    service_context = st.session_state.service_context

# Create the index from documents
if 'index' not in st.session_state:
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context)
    st.session_state.index = index
else:
    index = st.session_state.index

# Create the query engine
if 'query_engine' not in st.session_state:
    query_engine = index.as_query_engine(
        verbose=True, llm_predictor=llm_predictor)
    st.session_state.query_engine = query_engine
else:
    query_engine = st.session_state.query_engine
    
    
# Function to handle query
def query(payload):
    response = query_engine.query(payload["inputs"]["text"])
    # Transform response to string and remove
    response = str(response).lstrip("\n")
    # leading newline character if present
    return response

# Function to handle form callback


def form_callback():
    st.session_state.input_value = st.session_state.input

# Function to get text
def get_text(count):
    if count == 0:
        label = "Type a question about a movie: "
        value = "Who starred in the movie: Titanic?"
        input_text = st.text_input(
            label=label, value=value, key="input", on_change=form_callback)
    else:
        label = "User: "
        value = st.session_state.input_value
        input_text = st.text_input(label=label, value=value, key="input")
    return input_text


# User input
user_input = get_text(count=st.session_state.count)

# If user input is not empty, process the input
if user_input and user_input.strip() != '':
    output = query({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input,
        },
        "parameters": {"": ""},
    })

    # Increment count, append user input and generated output to session state
    st.session_state.count += 1
    st.session_state.past.append(user_input)
    if output:
        st.session_state.generated.append(output)

# If there are generated messages, display them
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')
