import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

import streamlit as st

# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"

# Store the API key in a variable.
OPENAI_API_KEY = "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"

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# Initialize the model.
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.0)

# Initialize an object for conversational memory.
buffer = ConversationEntityMemory(llm=llm)

# Create the chain for conversation, using a ConversationBufferMemory object.
conversation = ConversationChain(
    llm=llm, 
    memory=buffer, 
    #verbose=True, 
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE
)

# st.set_page_config(page_title="Site Seeing Demo", page_icon="📈")

# st.markdown("# Site Seeing Demo")
st.title('Site Seeing Recommendations')
st.sidebar.header("Site Seeing Demo")
st.write(
    """This demo illustrates how the app works with LLM to provide a site seeing itinerary! Enjoy!"""
)

# Title


col1, col2 = st.columns(2)
with col1:
    with st.form(key='form2'):
        st.header("Trip Details")
        city_input=st.text_input("What City?")
        days_input=st.number_input("How Many Days", step=1)
        activities_input=st.number_input("How Many Activies Per Day?", step=1)

        submit_button=st.form_submit_button(label="Submit")

with col2:
    with st.form(key='form3'):
        st.header("Trip Preferences")
        likes_one = st.radio("What is your TOP priority?", ("Music", "Food", "Museums", "Sports"))
        likes_two = st.multiselect("What else is important?", ["Adventure", "Food", "Michelin Star Meals", "Museums", "Nature", "Sports", "Wine Tasting"], key = "multi_input")
        likes_three = st.text_input("Any special requests?", key="special_input")

        submit_button=st.form_submit_button(label="Submit")

st.markdown("<hr>", unsafe_allow_html=True)

# Generate the query

st.title(f"Here is Your Itinerary")

# Generate the query
query = (
    f"""You are my personal travel agent. I'm planning a trip and would like you to prepare my
    itinerary based on the inputs from the submit_bottoms, including city, number of days
    nnumber of activities,top priority and trip preferences
    Please provide the itinerary in list format.
    When selecting daily activities, make sure that the site seeing activities are within 2 miles 
    of each other and make sure you include the correct city, number of days, and correct activities 
    per day as submitted in the form. For {days_input} in {city_input} with {activities_input} per day based on your top priority of {likes_one} for this trip and 
    the other things you like to do: {likes_two} And any special requests: {likes_three}. I will include {activities_input} per day."""
)

# Invoke the conversation chain
conversation.invoke(input=query)

# # Your prompt engineering goes here
# query = f"Please start the itinerary in {city_input} and provide an itinerary for the number of {days_input} with {activities_input} activities per day. Print this as bulleted items in a list under each day."
# conversation.invoke(input=query)

# query = f"Please spit out an itinerary for {activities_input} activities a day for the number of {days_input}. Please create a list by day starting with day 1."

# Predict the itinerary
itinerary = conversation.predict(input=query)

# Display the itinerary
st.write(itinerary)

# Ask the user if they like the itinerary
like_itinerary = st.radio("Do you like this itinerary?", options=["Yes", "No"])

st.markdown("<hr>", unsafe_allow_html=True)

# If the user likes the itinerary, display a message
if like_itinerary == "Yes":
    st.write("Great! I will provide a list of all the addresses.")

    query = (
        """please fill in the addresses and specific details for each of the activities in the itinerary"""
    )
    conversation.invoke(input=query)
    itinerary_details = conversation.predict(input=query)
    st.write(itinerary_details)

# If the user doesn't like the itinerary
if like_itinerary == "No":
    # Ask for feedback on what they didn't like
    feedback = st.text_area("What didn't you like about the itinerary?")
    
    # Add a button to submit feedback and request a new itinerary
    if st.button("Submit Feedback and Request New Itinerary"):
        # Clear the previous conversation history
        clear_history()
        
        # Provide confirmation message
        st.warning("Feedback submitted. Previous itinerary cleared. Please fill out the form again to request a new itinerary.")
