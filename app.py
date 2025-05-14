import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import (
    ArxivAPIWrapper, WikipediaAPIWrapper, GoogleSearchAPIWrapper, RedditAPIWrapper, YouTubeSearchAPIWrapper
)
from langchain_community.tools import (
    ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun, RedditQueryRun, YouTubeSearchTool
)
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize wrappers and tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

reddit_wrapper = RedditAPIWrapper()
reddit = RedditQueryRun(api_wrapper=reddit_wrapper)

youtube_wrapper = YouTubeSearchAPIWrapper()
youtube = YouTubeSearchTool(api_wrapper=youtube_wrapper)

# Streamlit app UI
st.title("🔎 LangChain - Chat with search")
st.write(
    """
    In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
    Now with Reddit and YouTube search! 🎥👾
    """
)

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.warning("Please enter your Groq API Key to proceed.")

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search Arxiv, Wikipedia, Reddit, YouTube and more. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user input
prompt = st.chat_input(placeholder="What is machine learning?")
if prompt and api_key:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LangChain tools
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki, reddit, youtube]

    # Agent with toolset
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    try:
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.write(response)
    except Exception as e:
        st.error("An error occurred. Please check your keys and try again.")
        st.write(str(e))

