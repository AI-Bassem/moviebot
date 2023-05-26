import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
import logging
import os
import sys
from pathlib import Path
import csv
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.agents import create_csv_agent
from dotenv import load_dotenv
from OctoAiCloudLLM import OctoAiCloudLLM
from langchain import HuggingFaceHub, OpenAI, PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import ( LLMPredictor, ServiceContext, download_loader, GPTVectorStoreIndex, LangchainEmbedding)
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(current_dir)
# Set logging level to CRITICAL
logging.basicConfig(level=logging.CRITICAL)
from streamlit_chat import message
import requests
import os
from OctoAiCloudLLM import OctoAiCloudLLM
st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
    )
os.environ["OCTOAI_API_TOKEN"] = st.secrets['OCTOAI_API_TOKEN']
os.environ["ENDPOINT_URL"] = st.secrets['ENDPOINT_URL']

os.environ["ENDPOINT_URL"] = 'https://dolly-demo-test-f1kzsig6xes9.octoai.cloud/predict'
os.environ["OCTOAI_API_TOKEN"] = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjNkMjMzOTQ5In0.eyJzdWIiOiI4NjBhYWY5NC00MTdmLTQ4MTAtOWVlMy1hMmYwN2MwNzNmZWEiLCJ0eXBlIjoidXNlckFjY2Vzc1Rva2VuIiwidGVuYW50SWQiOiI3M2UwNGZkMy0zMGRjLTQ3OWItYWYwYS01ZTViYzgwOGFjZTUiLCJ1c2VySWQiOiJkZGE5YTQxNC0yZWNhLTQxMWEtOGQ1Yy0zOTliN2NhYjAxOGMiLCJyb2xlcyI6WyJGRVRDSC1ST0xFUy1CWS1BUEkiXSwicGVybWlzc2lvbnMiOlsiRkVUQ0gtUEVSTUlTU0lPTlMtQlktQVBJIl0sImF1ZCI6IjNkMjMzOTQ5LWEyZmItNGFiMC1iN2VjLTQ2ZjYyNTVjNTEwZSIsImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkub2N0b21sLmFpIiwiaWF0IjoxNjg0Mzc4MTA2fQ.ClvQuflKpYu_h39YeEXRr9QY8vDwQNa9Ym7ZpSHFVv7SmhXgrZDXXZt_Y-8aHXUnmikWuGgd-_yOmob7IGKoc0YMvsLRjU64JXcO1SgRgV87asv1r2aUqkFrdYIHQzwwAxIDjMHQVoK7hWEJ6IZFwwoBxqucIAWNIpxueA_LOeBUn2q9ANOlICrmtr31uVkNskKQxB3Kekt09ut6uruNqGYFDSoDbELBMAQpLRGsutKYdqBWWl6mQCy_sMctNSp2Ccz6alcE-BtOUScvcfKUzvWQJAw1Ho6C5_4Cm6jcSTc4xO7AvmicuhBp-i91Z9a6lylfj4iZtMy0QUJuAC0vOw'


endpoint_url = os.getenv("ENDPOINT_URL")


st.header("Movie Bot - Demo")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

file = Path('rotten_tomatoes_top_movies.csv')

PagedCSVReader = download_loader("PagedCSVReader")

loader = PagedCSVReader()
documents = loader.load_data(file)
# Initialize the OctoAiCloudLLM
endpoint_url = os.getenv("ENDPOINT_URL")
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


if 'count' not in st.session_state:  # or st.session_state.count == 6:
    st.session_state.count = 0
    st.session_state.chat_history_ids = None
    st.session_state.old_response = ''

def query(payload):
	
    response = query_engine.query(payload["inputs"]["text"])
            #print('\n')

            # Transform response to string and remove leading newline character if present
    response = str(response).lstrip("\n")

    output = response
    

    #st.write("final result",str(output))
    return output

def form_callback():
    st.session_state.input_value = st.session_state.input
    
 
def get_text(count):
    if count==0:
        label="Type a question about a movie: "
        value="Who starred in Titanic?"
        input_text = st.text_input(label=label, value=value, key="input", on_change=form_callback)
    else:
        label="User: "
        value = st.session_state.input_value
        input_text = st.text_input(label=label, value=value, key="input")
    return input_text 



user_input = get_text(count=st.session_state.count)
#print(user_input)
#while user_input:
if user_input and user_input.strip() != '':
    #st.write(user_input)
    output = query({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input,
        },"parameters": {"repetition_penalty": 1.33},
    })

    st.session_state.count += 1

    st.session_state.past.append(user_input)
    if output:
        st.session_state.generated.append(output)
    #st.text_input("Bot: ", value=output["generated_text"], key="output", disabled=True)
if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

