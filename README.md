

# Movie Bot - Demo

## Requirements

- Streamlit

- OctoAI API access token  

- Endpoint URL for OctoAI API

## How to run

1\. Clone this repo

2\. Obtain an OctoAI API access token and endpoint URL

3\. Add the access token and endpoint URL as Streamlit secrets named `OCTOAI_API_TOKEN` and `ENDPOINT_URL` respectively

4\. Run `streamlit run app.py` to start the Streamlit app

5\. You can now chat with the bot and ask it questions about movies! For example:

- Who starred in the movie Titanic?

- What genre is The Matrix?

- When was Jurassic Park released?

 The bot will respond with the relevant information.

## How it works

This app uses the OctoAI Cloud LLM API to get responses from a large language model trained on movie data. Specifically:

- Movie data from Rotten Tomatoes Top Movies dataset is loaded as documents from the .csv file

- These documents are indexed using an OctoAI Embeddings endpoint running Instructor-Large model and GPTVectorStoreIndex from llama_index package

- User inputs are queried against this index using QueryEngine from llama_index

- The relevant text from the index is then sent to the OctoAI LLM endpoint using langchain

- Responses from the OctoAI API are displayed in the Streamlit chat interface

The full code and components used are available in `chatbot.py`. Please let me know if you have any questions!
