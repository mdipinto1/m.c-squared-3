import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

import streamlit as st


# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"

# # Store the API key in a variable.
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Opening Page",
    page_icon="👋",
)

st.image("/Users/christinekanouff/Desktop/MCSquare-P3-Streamlit/logo/TravelProLogo.png", width = 300)

st.write("# Welcome to TRAVEL PRO! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    TravelPro is an app in it's preliminary development, created by MCSquared.
    
    **👈 Select a demo from the sidebar** to see how it works.
    ### Want to learn more?  Talk to:
    - Matt DiPinto
    - Cindy Zhou
    - Christine Kanouff

"""
)