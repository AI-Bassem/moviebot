from langchain.document_loaders.csv_loader import CSVLoader
import logging
import os
import sys
from pathlib import Path
from getplot import getplot
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

# Load environment variables
load_dotenv()

# Set the file storage directory
FILES = "./files"


def init():
    """
    Initialize the files directory.
    """
    if not os.path.exists(FILES):
        os.mkdir(FILES)


def handle_exit():
    """
    Handle exit gracefully.
    """
    print("\nGoodbye!\n")
    sys.exit(1)

from pathlib import Path
from llama_index import download_loader


def ask(file):
    """
    Load the file, create the query engine and interactively answer user questions about the document.
    """
    print("Loading...")
    
    file = Path('rotten_tomatoes_top_movies.csv')
	#chunk the content and then convert it into document kind 
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
	# Load the PDFReader
    #loader = CSVLoader(file, source_column='title')
    #documents = loader.load()

    PagedCSVReader = download_loader("PagedCSVReader")

    loader = PagedCSVReader()
    documents = loader.load_data(file)
    # Initialize the OctoAiCloudLLM
    endpoint_url = os.getenv("ENDPOINT_URL")
    llm = OctoAiCloudLLM(endpoint_url=endpoint_url)
    llm_predictor = LLMPredictor(llm=llm)

    # Create the LangchainEmbedding
    embeddings = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    # Create the ServiceContext
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=1024, embed_model=embeddings)

    # Create the index from documents
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context)

    # Create the query engine
    query_engine = index.as_query_engine(
        verbose=True, llm_predictor=llm_predictor)

    # Clear the screen
    os.system("clear")

    print("Ready! Ask anything about movies")
    print('\n')
    print("Press Ctrl+C to exit")

    try:
        while True:
            prompt = input("\nPrompt: ")
            if prompt == "exit":
                handle_exit()
            '''
            #build the vector store
            #embeddings = OpenAIEmbeddings()
            docsearch = FAISS.from_documents(documents, embeddings)
            # ref: https://github.com/hwchase17/langchain/issues/2255
            retriever = docsearch.as_retriever(search_kwargs={"k": 1})
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)
            '''
            response = query_engine.query(prompt)
            #print('\n')

            # Transform response to string and remove leading newline character if present
            response = str(response).lstrip("\n")

            print("Response: " + response)
    except KeyboardInterrupt:
        handle_exit()


def select_file():
    """
    Select a file for processing.
    """
    os.system("clear")
    files = [file for file in os.listdir(FILES) if file.endswith(".pdf")]

    if not files:
        return "file.pdf" if os.path.exists("file.pdf") else None

    print("Select a file")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")
    print()

    try:
        possible_selections = [i for i in range(len(files) + 1)]
        selection = int(input("Enter a number, or 0 to exit: "))

        if selection == 0:
            handle_exit()
        elif selection not in possible_selections:
            select_file()
        else:
            file_path = os.path.abspath(
                os.path.join(FILES, files[selection - 1]))

        return file_path
    except ValueError:
        return select_file()
    
if __name__ == "__main__":
    # Initialize the file directory
    init()
    # Prompt user to select a file
    file = select_file()
    if file:
        # Start the interactive query session
        ask(file)
    else:
        print("No files found")
        handle_exit()
