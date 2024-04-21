import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

load_dotenv()

# Store the API key in a variable.
OPENAI_API_KEY = "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"


# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"

# Initialize the model.
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.0)

# Initialize an object for conversational memory.
buffer = ConversationEntityMemory(llm=llm)

# Create the chain for conversation, using a ConversationBufferMemory object.
conversation = ConversationChain(
    llm=llm, 
    memory=buffer, 
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE
)


# Define the function for generating itinerary based on user inputs
def generate_itinerary(City, Days, Activities_Per_Day, Top_Priority, Other_Preferences, Special_Requests):
    query = (
        f"""You are my personal travel agent. I'm planning a trip and would like you to prepare my
        itinerary for {Days} days in {City} with {Activities_Per_Day} activities per day.
        My top priority for this trip is {Top_Priority}, and I also like to do {', '.join(Other_Preferences)}
        with any special requests: {Special_Requests}. Please provide the itinerary by day in list format."""
    )
    
    conversation.invoke(input=query)
    itinerary = conversation.predict(input=query)
    return itinerary

# Create the Gradio interface
app=gr.Interface(
    fn=generate_itinerary,
    inputs=[
        gr.Textbox(label="City"),
        gr.Number(label="Days", step=1),
        gr.Number(label="Activities Per Day", step=1),
        gr.Radio(label="Top Priority", choices=["Music", "Food", "Museums", "Sports"]),
        gr.CheckboxGroup(label="Other Preferences", choices=["Adventure", "Food", "Michelin Star Meals", "Museums", "Nature", "Sports", "Wine Tasting"]),
        gr.Textbox(label="Special Requests"),
    ],
    outputs=gr.Textbox(label="Itinerary"),
    title="Site Seeing Recommendations",
    description="This demo illustrates how the app works with LLM to provide a site seeing itinerary! Enjoy!",
)
app.launch()