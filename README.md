In this tutorial, we'll build a chat app that lets a user ask questions about [Rotten Tomatoes Top Movies Dataset](https://www.kaggle.com/datasets/thedevastator/rotten-tomatoes-top-movies-ratings-and-technical).

# Preliminaries

This walkthrough assumes you already have some basic familiarity with the OctoAI Compute Service. If you haven't already read the [Getting Started](https://docs.octoai.cloud/docs/getting-started) section of our docs, we highly recommend you do so now. If this is your first time building an end-to-end Generative AI application, check out the Stable Diffusion example.

- [OctoAI compute service](https://octoai.cloud/): The OctoAI compute service provides fast and simple Machine Learning inference capability for us. In general, it can turn any container or Python code into a production-grade endpoint in minutes.
- [LLama2-13B ](https://huggingface.co/meta-llama/Llama-2-13b): This is the open-source transformer model at the heart of our chat app. 
- [Streamlit](https://github.com/streamlit): A tool for building lightweight, beautiful, and shareable Python-based web applications.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html): A useful package and environment manager that includes both Python and pip out of the box. The standard Python distribution includes each of the following libraries that we’ll leverage in our application:
  - [Requests](https://requests.readthedocs.io/en/latest/): Allows us to send and receive HTTP requests and responses using Python code. We use this library to make it easy to access the OctoML compute service; we simply provide some basic information including the URL and the stable diffusion parameters that we want to send.
  - [Langhchain](https://python.langchain.com/en/latest/index.html): A popular library for building LLM applications, Langchain extends LLM capabilities through the use of constructs called `Agents` and `Chains`. Used in concert, these constructs allow us to build more complex applications that can be deployed to production.


# 💻  Create a New Virtual Environment & Install Streamlit

1. Install the appropriate version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS.
2. From the terminal, create a conda environment using the`conda create --name moviebotapp`.
3. Activate the conda environment with `conda activate moviebotapp`.

# 🧑‍💻 Build the MovieBot Application

Like before, we will be using Streamlit to run our chat application. The code for this application is in the `moviebot` [repo](https://github.com/AI-Bassem/moviebot). 

- Clone this repo to your local machine using the`git clone`.
- Install application dependencies with `pip install -r requirements.txt`.
- Create a new `OCTOAI_API_TOKEN`. You can create new keys under "Settings" (the gear icon to the left) in the OctoAI dashboard.
- Be sure your `OCTOAI_API_TOKEN` is set in your `'/.streamlit/secrets.toml'`file as `OCTOAI_API_TOKEN="<paste token here>"`
- Add your ENDPOINT_URL to the '`/.streamlit/secrets.toml'`file as `ENDPOINT_URL="<paste url here>"`. 

# Deploy and run the MovieBot App

Now that you have built your application, you can deploy it locally within your conda environment using the pre-built script that we’ve provided. Here’s how:

1. From your terminal window, `cd` Into `/moviebotapp` folder where `streamlit_app.py` is located
2. Run this command: `streamlit run chatbot.py`
3. Open a browser window and navigate to `localhost:8501` to see your application running.
4. You can now chat with the bot and ask it questions about movies! For example:
   - Who starred in the movie Titanic?
   - What genre is The Matrix?
   - When was Jurassic Park released?

# How it works:

- Rotten Tomatoes movie data is loaded from the `.csv` file. 
- The loaded docs are indexed using Instructor-Large model and FAISS from Langchain package.
- User inputs are queried against this index using FAISS from Langchain. 
- The relevant text from the index is then sent to the OctoAI LLM endpoint using langchain. 
- Responses from the OctoAI API are displayed in the Streamlit chat interface
