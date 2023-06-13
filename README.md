# Building a Moviebot Chat App 
In this tutorial, we'll build a chat app that let's a user ask questions about Rotten Tomatoes Top Movies Database.

# Preliminaries
This walkthrough assumes you already have some basic familiarity with the OctoAI Compute Service. If you haven't already read the [Getting Started](https://docs.octoai.cloud/docs/getting-started) section of our docs, we highly recommend you do so now. If this is your first time building an end-to-end Generative AI application, check out the Stable Diffusion example.

- [OctoAI compute service](https://octoai.cloud/): The OctoAI compute service provides fast and simple Machine Learning inference capability for us. In general, it can turn any container or Python code into a production-grade endpoint in minutes.
- [MPT-7B ](https://huggingface.co/mosaicml/mpt-7b): This is the open-source transformer model at the heart of our chat app. 
- [Streamlit](https://github.com/streamlit): A tool for building lightweight, beautiful, and shareable Python-based web applications.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html): A useful package and environment manager that includes both Python and pip out of the box. The standard Python distribution includes each of the following libraries that we’ll leverage in our application:
    - [Requests](https://requests.readthedocs.io/en/latest/): Allows us to send and receive HTTP requests and responses using Python code. We use this library to make it easy to access the OctoML compute service; we simply provide some basic information including the URL and the stable diffusion parameters that we want to send.
    - [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/): A "data framework" for LLMs that allows us to do things like connect to documents and databases, structure our data for LLMs, run queries, and integrate with popular applications and frameworks.
    - [Langhchain](https://python.langchain.com/en/latest/index.html): A popular library for building LLM applications, Langchain extends LLM capabilities through the use of constructs called `Agents` and `Chains`. Used in concert, these constructs allow us to build more complex applications that can be deployed to production.

# 🧑‍💻 Step 1: Create a new OctoAI Endpoint from a Template
* Go to [OctoAI](https://octoai.cloud/), click `Endpoints`, and select "Chatbot (Research-Only)" from the template cards to use the [mpt-7b-demo](https://octoai.cloud/templates/mpt-7b-demo) template.
* You can experiment with the conversational outputs of this LLM by typing a prompt into the `Prompt1` field and clicking `Generate`.
* When you're ready to build your own production-grade chatbot, click the `Clone` button below the cURL Example to create a new endpoint from this template.
* Copy the `Endpoint URL`

# 💻 Step 2: Create a New Virtual Environment & Install Streamlit
1. Install the appropriate version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS.
2. From the terminal, create a conda environment using `conda create --name moviebotapp`.
3. Activate conda environment with `conda activate moviebotapp`.

# 🧑‍💻 Step 3: Build the MovieBot Application
Like before, we will be using Streamlit to run our chat application. The code for this application is in the `/moviebot` folder of this repo. 
* Clone this repo to your local machine using `git clone`.
* Install application dependencies with `pip install -r requirements.txt`.
* Create a new `OCTOAI_API_TOKEN`. You can create new keys under "Settings" (the gear icon to the left) in the OctoAI dashboard.
* Be sure your `OCTOAI_API_TOKEN` is set in your '/.streamlit/secrets.toml' file as `OCTOAI_API_TOKEN="<paste token here>"`
* Add your ENDPOINT_URL to the '/.streamlit/secrets.toml' file as `ENDPOINT_URL="<paste url here>"`. 



# 🚢 Step 3: Deploy and run the MovieBot App
Now that you have built your application, you can deploy it locally within your conda environment using the pre-built script that we’ve provided. Here’s how:
1. From your terminal window, `cd` into `/moviebotapp` folder where `streamlit_app.py` is located
2. Run this command: `streamlit run chatbot.py`
3. Open a browser window and navigate to `localhost:8501` to see your application running.
4. You can now chat with the bot and ask it questions about movies! For example:
    * Who starred in the movie Titanic?
    * What genre is The Matrix?
    * When was Jurassic Park released?

# How it works: 
* Rotten Tomatoes movie data is loaded from the `.csv` file. 
* The loaded docs are indexed using Instructor-Large model and GPTVectorStoreIndex from llama_index package.
* User inputs are queried against this index using QueryEngine from LlamaIndex. 
* The relevant text from the index is then sent to the OctoAI LLM endpoint using langchain. 
* Responses from the OctoAI API are displayed in the Streamlit chat interface

# 📚 Next Steps
If you have any questions or comments, please check our [docs](https://docs.octoai.cloud/docs) or reach out and ask a question directly in our [discord community](https://discord.com/invite/rXTPeRBcG7)!  We look forward to seeing what you build next!

Thanks for using OctoAI!
