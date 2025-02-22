# m.c-squared-3




Sight Seeing Portion of Application:  

This application is an example of how advanced language models can be integrated into practical applications, like travel planning, using modern Python libraries for natural language processing and web interface design. I originally used Streamlit for this portion of the app.  I have included those files in a separate file if you want to compare Streamlit to Gradio.

Here's the Environment set up:
Openai API Key necessary to run
OPENAI_MODEL = gpt-3.5-turbo

LangChain and Model Initialization:  llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.0): Initializes the LangChain wrapper for the OpenAI chat model, setting the temperature to 0.0 for deterministic outputs.

buffer = ConversationEntityMemory(llm=llm): Initializes a conversation memory buffer, which helps the model maintain context or remember aspects of the conversation.

Conversation Chain Setup:  conversation = ConversationChain(llm=llm, memory=buffer, prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE): Sets up a conversation chain with memory capabilities, which processes user inputs and generates responses using a structured template.

Gradio Interface and Application Logic:  The function generate_itinerary is defined to create a personalized travel itinerary based on parameters like city, number of days, activities per day, and user preferences. The Gradio library is used to build a user-friendly web interface (gr.Interface). 

The interface includes text boxes, number inputs, radio buttons, and checkboxes for users to input their preferences. The itinerary generator function is connected to this interface, allowing users to get real-time itineraries generated by the model.  Launching the Interface:  app.launch(): This command launches the Gradio web interface, making the application accessible via a web browser. 
