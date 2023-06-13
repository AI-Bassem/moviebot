# Import the required libraries
import os
import dill
import streamlit as st
from pathlib import Path
from streamlit_chat import message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.agents import create_csv_agent
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from octoai_endpoint import OctoAIEndpoint
from octoai_embeddings import OctoAIEmbeddings
from llama_index import (LLMPredictor, ServiceContext,
                         download_loader, GPTVectorStoreIndex, LangchainEmbedding)

# Set the page configurations
st.set_page_config(page_title="​🎬  IMDBot - Powered by OctoAI",
                   page_icon=":robot:")

# Set up environment variables


def setup_env_variables():
    """Set up environment variables."""
    os.environ["OCTOAI_API_TOKEN"] = st.secrets['OCTOAI_API_TOKEN']
    os.environ["ENDPOINT_URL"] = st.secrets['ENDPOINT_URL']

# Initialize session state


def handle_session_state():
    """Initialize the session state if not already done."""
    st.session_state.setdefault('generated', [])
    st.session_state.setdefault('past', [])
    st.session_state.setdefault('q_count', 0)

# Load movie data


def load_data(file_path):
    """Load movie data from a CSV file."""
    PagedCSVReader = download_loader("PagedCSVReader")
    loader = PagedCSVReader()
    return loader.load_data(file_path)

# Initialize the OctoAiCloudLLM and LLMPredictor


def initialize_llm(endpoint_url):
    """Initialize the OctoAiCloudLLM and LLMPredictor."""
    llm = OctoAIEndpoint(endpoint_url=endpoint_url, model_kwargs={
                         "max_new_tokens": 200, "temperature": 0.75, "top_p": 0.95, "repetition_penalty": 1, "seed": None, "stop": [], })
    return LLMPredictor(llm=llm)

# Create LangchainEmbedding


def create_embeddings():
    """Create and return LangchainEmbedding instance."""
    if 'embeddings' not in st.session_state:
        embeddings = LangchainEmbedding(OctoAIEmbeddings(
            endpoint_url="https://instruct-f1kzsig6xes9.octoai.cloud/predict"))
        st.session_state['embeddings'] = embeddings
    return st.session_state['embeddings']

# Create ServiceContext


def create_service_context(llm_predictor, embeddings):
    """Create and return ServiceContext instance."""
    if 'service_context' not in st.session_state:
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, chunk_size_limit=400, embed_model=embeddings)
        st.session_state['service_context'] = service_context
    return st.session_state['service_context']

# Create Index


def create_index(documents, service_context):
    """Create and return GPTVectorStoreIndex instance."""
    if 'index' not in st.session_state:
        path = Path("index.pkl")
        if path.exists():
            index = dill.load(open(path, "rb"))
        else:
            index = GPTVectorStoreIndex.from_documents(
                documents, service_context=service_context)
            #dill.dump(index, open(path, "wb")) #https://github.com/jerryjliu/llama_index/issues/886
        st.session_state['index'] = index
    return st.session_state['index']

# Create Query Engine


def create_query_engine(index, llm_predictor):
    """Create and return a query engine instance."""
    if 'query_engine' not in st.session_state:
        query_engine = index.as_query_engine(
            verbose=True, llm_predictor=llm_predictor)
        st.session_state['query_engine'] = query_engine
    return st.session_state['query_engine']

# Process Query


def query(payload, query_engine):
    """Process a query and return a response."""
    response = query_engine.query(payload["inputs"]["text"])
    # Transform response to string and remove leading newline character if present
    return str(response).lstrip("\n")

# Handle Form Callback


def form_callback():
    """Handle the form callback."""
    st.session_state['input_value'] = st.session_state['input']

# Get Text


def get_text(q_count):
    """Display a text input field on the UI and return the user's input."""
    label = "Type a question about a movie: "
    value = "Who directed the movie Jaws?\n"
    return st.text_input(label=label, value=value, key="input", on_change=form_callback)


def main():
    # Setup the environment variables
    setup_env_variables()
    # Set the endpoint url
    endpoint_url = os.getenv("ENDPOINT_URL")
    # Initialize the session state
    handle_session_state()
    # Load the data
    documents = load_data(Path('rotten_tomatoes_top_movies.csv'))
    # Initialize the LLM predictor
    llm_predictor = initialize_llm(endpoint_url)
    # Create the embeddings
    embeddings = create_embeddings()
    # Create the service context
    service_context = create_service_context(llm_predictor, embeddings)
    # Create the index
    index = create_index(documents, service_context)
    # Create the query engine
    query_engine = create_query_engine(index, llm_predictor)
    # Display the header
    st.subheader("​🎬  IMDBot - Powered by OctoAI")
    st.markdown('* :movie_camera: Tip #1: IMDBot is great at answering factual questions like: "Who starred in the Harry Potter movies?" or "What year did Jaws come out?')
    st.markdown('* :black_nib: Tip #2: IMDBot loves the word "synopsis" -- we suggest using it if you are looking for plot summaries. Otherwise, expect some hallucinations.')
    st.markdown("* :blush: Tip #3: IMDbot has information about 500 popular movies, but is not comprehensive. It probably won't know some more obscure films.")
    st.markdown("### Welcome to the IMDBot demo")
    st.sidebar.image("octoml-octo-ai-logo-color.png", caption="Try OctoML's new compute service for free by signing up for early access: https://octoml.ai/")

    try:
        # Get the user input
        user_input = get_text(q_count=st.session_state['q_count'])
        # If user input is not empty, process the input
        if user_input and user_input.strip() != '':
            output = query({"inputs": {"text": user_input, }}, query_engine)
            # Increment q_count, append user input and generated output to session state
            st.session_state['q_count'] += 1
            st.session_state['past'].append(user_input)
            if output:
                st.session_state['generated'].append(output)
        # If there are generated messages, display them
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state['past'][i],
                        is_user=True, key=f'{str(i)}_user')
                message(st.session_state["generated"][i], key=str(i))

    except Exception as e:
        st.error("Something went wrong. Please try again.")


if __name__ == "__main__":
    main()
