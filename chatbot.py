# Import the required libraries
import os
import pickle
import streamlit as st
from pathlib import Path
from streamlit_chat import message
from langchain.llms.octoai_endpoint import OctoAIEndpoint
from langchain.embeddings.octoai_embeddings import OctoAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.faiss import FAISS
from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Set the page configurations
st.set_page_config(page_title="​🎬  IMDBot - Powered by OctoAI", page_icon=":robot:")


# Initialize session state


def handle_session_state():
    """Initialize the session state if not already done."""
    st.session_state.setdefault("generated", [])
    st.session_state.setdefault("past", [])
    st.session_state.setdefault("q_count", 0)


# Set up environment variables


def setup_env_variables():
    """Set up environment variables."""
    OCTOAI_API_TOKEN = st.secrets.get("OCTOAI_API_TOKEN", "")
    os.environ["OCTOAI_API_TOKEN"] = OCTOAI_API_TOKEN

    ENDPOINT_URL = st.secrets.get(
        "ENDPOINT_URL", "https://text.octoai.run/v1/chat/completions"
    )
    os.environ["ENDPOINT_URL"] = ENDPOINT_URL


# Load movie data


def load_data(file_path):
    """Load movie data from a CSV file."""
    loader = CSVLoader(
        file_path=file_path, encoding="utf-8", csv_args={"delimiter": ","}
    )
    return loader.load()


# Initialize the OctoAiCloudLLM and LLMPredictor


def initialize_llm(endpoint_url):
    """Initialize the OctoAiCloudLLM and LLMPredictor."""
    llm = OctoAIEndpoint(
        endpoint_url=endpoint_url,
        model_kwargs={
            "model": "llama-2-13b-chat-fp16",
            "messages": [
                {
                    "role": "system",
                    "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                }
            ],
            "stream": False,
            "max_tokens": 256,
        },
    )

    return llm  # llms.LangChainLLM(llm=llm)


# Create LangchainEmbedding


def create_embedding(documents):
    """Create and return LangchainEmbedding instance."""
    if "embeddings" not in st.session_state:
        octo_embed = OctoAIEmbeddings(
            octoai_api_token=os.environ["OCTOAI_API_TOKEN"],
            endpoint_url="https://instructor-large-f1kzsig6xes9.octoai.run/predict",
        )
        st.session_state["embeddings"] = octo_embed
    return st.session_state["embeddings"] 

# Create Index


def create_index(documents, embedding):
    """Create and return db store."""
    if "index" not in st.session_state:
        path = Path("index.pkl")
        if path.exists():
            index = pickle.load(open(path, "rb"))
        else:
            
            index = FAISS.from_documents(
                documents,
                embedding,
            )

            pickle.dump(
                index, open(path, "wb")
            )
        st.session_state["index"] = index
    return st.session_state["index"]


# Create Query Engine
from llama_index import indices, query_engine, response_synthesizers, retrievers


def create_query_engine(index):
    """Create and return a query engine instance."""

    if "query_engine" not in st.session_state:
        retriever = index.as_retriever()
        st.session_state["query_engine"] = retriever
    return st.session_state["query_engine"]


# Process Query


def query(payload, llm, retriever):
    """Process a query and return a response."""
    query = payload["inputs"]["text"]

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = llm

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke(query)
    # Transform response to string and remove leading newline character if present
    return str(response).lstrip("\n")


# Handle Form Callback


def form_callback():
    """Handle the form callback."""
    st.session_state["input_value"] = st.session_state["input"]


# Get Text


def get_text(q_count):
    """Display a text input field on the UI and return the user's input."""
    label = "Type a question about a movie: "
    value = "Who directed the movie Jaws?\n"
    return st.text_input(label=label, value=value, key="input", on_change=form_callback)


import traceback


def main():
    st.subheader("​🎬  IMDBot - Powered by OctoAI")
    st.markdown(
        '* :movie_camera: Tip #1: IMDBot is great at answering factual questions like: "Who starred in the Harry Potter movies?" or "What year did Jaws come out?'
    )
    st.markdown(
        '* :black_nib: Tip #2: IMDBot loves the word "synopsis" -- we suggest using it if you are looking for plot summaries. Otherwise, expect some hallucinations.'
    )
    st.markdown(
        "* :blush: Tip #3: IMDbot has information about 500 popular movies, but is not comprehensive. It probably won't know some more obscure films."
    )
    st.markdown("### Welcome to the IMDBot demo")
    st.sidebar.image(
        "octoml-octo-ai-logo-color.png",
        caption="Try OctoML's new compute service for free by signing up here: https://octoml.ai/",
    )
    # Setup the environment variables
    setup_env_variables()
    # Display the header
    # Set the endpoint url
    endpoint_url = os.getenv("ENDPOINT_URL")

    try:
        # Initialize the session state
        handle_session_state()
        # Load the data
        documents = load_data(Path("rotten_tomatoes_top_movies.csv"))
        # Create the embeddings
        embedding_model = create_embedding(documents)
        # Create the index
        index = create_index(documents, embedding_model)
        # Initialize the LLM predictor
        llm_predictor = initialize_llm(endpoint_url)
        # Create the query engine
        query_engine = create_query_engine(index)
        # Get the user input
        user_input = get_text(q_count=st.session_state["q_count"])
        # If user input is not empty, process the input
        if user_input and user_input.strip() != "":
            output = query(
                {
                    "inputs": {
                        "text": user_input,
                    }
                },
                llm_predictor,
                query_engine,
            )
            # Increment q_count, append user input and generated output to session state
            st.session_state["q_count"] += 1
            st.session_state["past"].append(user_input)
            if output:
                st.session_state["generated"].append(output)
        # If there are generated messages, display them
        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["past"][i], is_user=True, key=f"{str(i)}_user")
                message(st.session_state["generated"][i], key=str(i))

    except Exception as e:
        tb = traceback.format_exc()

        # Display the error message with traceback
        st.error(f"Something went wrong: {e}\n\nTraceback:\n{tb} Please try again.")


if __name__ == "__main__":
    main()
