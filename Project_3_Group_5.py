import pandas as pd
import json
import numpy as np
# import cudf as pd
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import cosine_similarity
from geopy import distance
import gradio as gr
import matplotlib as plt
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.0)
buffer = ConversationEntityMemory(llm=llm)
conversation = ConversationChain(llm=llm, memory=buffer, prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE)

# Define the function for generating itinerary based on user inputs
def generate_itinerary(City, Days, Activities_Per_Day, Top_Priority, Other_Preferences, Special_Requests):
    query = (
        f"""You are my personal travel agent. I'm planning a trip and would like you to prepare my
        itinerary for {Days} days in {City} with {Activities_Per_Day} activities per day.
        My top priority for this trip is {Top_Priority}, and I also like to do {', '.join(Other_Preferences)}
        with any special requests: {Special_Requests}. Please provide the itinerary by day in list format and include
        the addresses of each of the activities."""
    )
    itinerary = conversation.predict(input=query)
    return itinerary

def feedback_itinerary(user_feedback):
    query = f"I didn't like that itinerary because {user_feedback}. Please provide me a new itinerary based on the new feedback. Keep the original inputs."
    new_itinerary = conversation.predict(input=query)
    return new_itinerary

def update_visibility(radio_input):
    if radio_input == 'Yes':
        return gr.Textbox(visible=False), gr.Button(visible=False)
    else:
        return gr.Textbox(visible=True), gr.Button(visible=True)    












business_df = pd.read_json('./Data/yelp_academic_dataset_business.json', lines=True)
business_df = business_df[(business_df['state'] == 'TN') & (business_df['is_open'] == 1)]
business_df['categories'] = business_df['categories'].fillna('')
business_df = business_df[business_df['categories'].str.contains('Restaurants')]

photos_df = pd.read_json('./Data/photos.json', lines=True)
photos_df = photos_df.loc[photos_df['business_id'].isin(business_df['business_id'])]

checkin_df = pd.read_json('./Data/yelp_academic_dataset_checkin.json', lines=True)
checkin_df = checkin_df.loc[checkin_df['business_id'].isin(business_df['business_id'])]

useful_chunks = []

for chunk in pd.read_json('./Data/yelp_academic_dataset_review.json', lines=True, chunksize=100000):
    filtered_chunk = chunk.loc[chunk['business_id'].isin(business_df['business_id'])]
    useful_chunks.append(filtered_chunk)

reviews_df = pd.concat(useful_chunks)
reviews_df.reset_index(drop=True, inplace=True)

tip_df = pd.read_json('./Data/yelp_academic_dataset_tip.json', lines=True)
tip_df = tip_df.loc[tip_df['business_id'].isin(business_df['business_id'])]

summary_reviews_df = pd.read_csv('./Data/nashville_business_reviews_summary.csv', sep='|')

def integrate_reviews(row, sentiment):
    try:
        review = summary_reviews_df.loc[(summary_reviews_df['sentiment'] == sentiment) & (summary_reviews_df['business_id'] == row['business_id'])]['summary'].values[0]
    except:
        review = 'No Reviews'
    return review

business_df['negative_summary'] = business_df.apply(integrate_reviews,axis=1, args=('negative',))
business_df['positive_summary'] = business_df.apply(integrate_reviews,axis=1, args=('positive',))

all_restaurant_types = business_df['categories'].str.split(',').explode().str.strip().value_counts().index
valid_types = all_restaurant_types[:128].tolist()
types_to_remove = ['Restaurants','Event Planning & Services','Caterers','Music Venues','Food Delivery Services','Venues & Event Spaces','Hotels & Travel','Convenience Stores','International Grocery','Performing Arts','Florists','Active Life','Food','Nightlife', 'Arcades', 'Flowers & Gifts','Butcher', 'Jazz & Blues','Party & Event Planning','Dance Clubs', "Arts & Entertainment", "Shopping", "Ethnic Food", "Street Vendors",
    "Karaoke", "Pasta Shops", "Meat Shops", "Pop-Up Restaurants", "Farmers Market","Automotive"]
for type in types_to_remove:
    valid_types.remove(type)

business_df.dropna(subset=['attributes'], inplace=True)

business_df['OutdoorSeating'] = business_df['attributes'].apply(lambda x: x.get('OutdoorSeating', None))
business_df['Alcohol'] = business_df['attributes'].apply(lambda x: x.get('Alcohol', None))
business_df['RestaurantsPriceRange2'] = business_df['attributes'].apply(lambda x: x.get('RestaurantsPriceRange2', None))


business_df['OutdoorSeating'].fillna(False, inplace=True)
business_df['OutdoorSeating'].replace({'False': False, 'True': True, 'None': False}, inplace=True)

business_df['Alcohol'].fillna('none', inplace=True)
business_df['Alcohol'].replace({
                            "u'none'" : 'none',
                            "u'full_bar'" : 'full_bar',
                            "u'beer_and_wine'" : 'beer_and_wine',
                            "'none'" : 'none',
                            "'full_bar'" : 'full_bar',
                            "'beer_and_wine'" : 'beer_and_wine',
                            }, inplace=True)

business_df['RestaurantsPriceRange2'].fillna(2, inplace=True)
business_df['RestaurantsPriceRange2'] = business_df['RestaurantsPriceRange2'].astype(int)


business_df['hours'].fillna("{'Monday': '0:0-0:0', 'Tuesday': '0:0-0:0', 'Wednesday': '0:0-0:0', 'Thursday': '0:0-0:0', 'Friday': '0:0-0:0', 'Saturday': '0:0-0:0', 'Sunday': '0:0-0:0'}", inplace=True)


def encode_top_categories(row, valid_types):
    row_categories = set(row['categories'])
    return [1 if cat in row_categories else 0 for cat in valid_types]


business_df['categories'] = business_df['categories'].str.split(',')
business_df['categories'] = business_df['categories'].apply(lambda x: [str(cat).strip() for cat in x])

mlb = MultiLabelBinarizer(classes=valid_types)
encoded_array = mlb.fit_transform(business_df['categories'])
encoded_df = pd.DataFrame(encoded_array, columns=mlb.classes_, index=business_df.index)

business_df = pd.concat([business_df, encoded_df], axis=1)
business_df = pd.get_dummies(business_df, columns=['Alcohol', 'OutdoorSeating', 'RestaurantsPriceRange2'], dtype=int)

scaler = MinMaxScaler()
scaler.fit(business_df[['stars']])
business_df['stars_scaled'] = scaler.transform(business_df[['stars']])

american_cuisine = [
    "American (Traditional)", "American (New)", "Burgers", "Barbeque",
    "Southern", "Steakhouses", "Comfort Food", "Cajun/Creole", "Hot Dogs", 
    "New Mexican Cuisine"
]

international_cuisine = [
    "Mexican", "Tex-Mex", "Italian", "Chinese", "Japanese", "Sushi Bars",
    "Asian Fusion", "Mediterranean", "Greek", "Thai", "Latin American",
    "Middle Eastern", "Indian", "Vietnamese", "French", "Korean", "Spanish",
    "Turkish", "Caribbean", "Ramen", "Salvadoran", "Poke", "Hawaiian",
    "Laotian", "Halal", "Ethiopian", "African"
]

fast_food_casual = [
    "Fast Food", "Sandwiches", "Pizza", "Chicken Wings", "Tacos", "Diners",
    "Food Trucks", "Hot Dogs", "Fish & Chips", "Donuts", "Waffles", "Acai Bowls",
    "Wraps", "Cheesesteaks", "Food Court"
]

bars_nightlife = [
    "Bars", "Cocktail Bars", "Sports Bars", "Pubs", "Lounges", "Dive Bars",
    "Wine Bars", "Beer Bar", "Tapas/Small Plates", "Gastropubs", "Breweries",
    "Brewpubs", "Beer Gardens", "Whiskey Bars", "Hookah Bars"
]

health_specialty_foods = [
    "Salad", "Vegetarian", "Vegan", "Gluten-Free", "Juice Bars & Smoothies",
    "Health Markets"
]

beverages = [
    "Coffee & Tea", "Specialty Food", "Wine & Spirits", "Beer", "Coffee Roasteries",
    "Bubble Tea"
]

desserts_bakeries = [
    "Desserts", "Ice Cream & Frozen Yogurt", "Bakeries", "Creperies"
]

cultural_local_flavors = [
    "Local Flavor", "Soul Food"
]

list_of_cats = {
    'American':american_cuisine,
    'International':international_cuisine,
    'Health Food':health_specialty_foods,
    'Local and Cultural':cultural_local_flavors,
    'Fast Food':fast_food_casual,
    'Coffee and Beverages':beverages,
    'Dessert':desserts_bakeries,
    'Bars and Nightlife':bars_nightlife,
}

price_dict = {
    '$':'RestaurantsPriceRange2_1',
    '$$':'RestaurantsPriceRange2_2',
    '$$$':'RestaurantsPriceRange2_3',
    '$$$$':'RestaurantsPriceRange2_4'
}

bar_dict = {
    "Beer and Wine":'Alcohol_beer_and_wine',
    "Full Bar":'Alcohol_full_bar',
    "None":'Alcohol_none'
}


X_list = business_df.columns[16:]
X = business_df[X_list]
y = business_df['name']


user_dict = {}
for column in X_list:
    user_dict[column] = 0

user_vector = pd.Series(user_dict).values.reshape(1,-1)


width, height = 200, 200  
background_color = 'grey'  
text = 'No Image Found'  
font_color = 'white'  
no_photos_img = Image.new('RGB', (width, height), color = background_color)
draw = ImageDraw.Draw(no_photos_img)
font = ImageFont.load_default()
text_width, text_height = 75, 75
x = (width - text_width) / 2
y = (height - text_height) / 2
draw.text((x, y), text, font=font, fill=font_color)


weight_factor = 2.25
rating_factor = 1.25
weights_array = np.ones_like(user_vector)
weights_array[:,:-10] = weight_factor
weights_array[:,-1] = rating_factor


def restaurant_distances(row,lat,long):
    return round(distance.distance((lat,long),(row['latitude'],row['longitude'])).miles,1)

def populate_range_df(airbnb_lat, airbnb_long):
    business_df['airbnb_range'] = business_df.apply(restaurant_distances, axis=1, args=(airbnb_lat, airbnb_long))

def reset_user():
    global user_dict
    user_dict = {}
    for column in X_list:
        user_dict[column] = 0
    user_dict['stars_scaled'] = rating_factor
    print(pd.Series(user_dict).values.reshape(1,-1))

def calculate_best_restaurant(choice, option, price_range, indoor_outdoor, drinks, distance):
    user_dict['stars_scaled'] = rating_factor
    search_df = business_df.loc[business_df['airbnb_range'] <= distance].copy()

    if choice == 'Category':
        for cuisine in list_of_cats[option]:
            user_dict[cuisine] = weight_factor
    else:
        user_dict[option] = weight_factor

    
    for price in price_dict.keys():
        if price in price_range:
            user_dict[price_dict[price]] = 1
    
    if indoor_outdoor == 'Outdoor Seating':
        user_dict['OutdoorSeating_True'] = 1

    if 'None' in drinks:
        user_dict['Alcohol_none'] = 1
    elif "Doesn't Matter" in drinks:
        user_dict['Alcohol_beer_and_wine'] = 1
        user_dict['Alcohol_full_bar'] = 1
        user_dict['Alcohol_none'] = 1
    else:
        for bar_type in drinks:
            user_dict[bar_dict[bar_type]] = 1
        
    
    similarities = cosine_similarity(search_df[X_list].values*weights_array, pd.Series(user_dict).values.reshape(1,-1))
    print(pd.Series(user_dict).values.reshape(1,-1))
    search_df['Match'] = similarities
    search_df.sort_values(by='Match', inplace=True, ascending=False)

    business_id1, business_id2, business_id3 = search_df.head(3)['business_id'].values.tolist()
    search_df.set_index('business_id', drop=True, inplace=True)
    search_df.rename(columns={'name':'english_name'}, inplace=True)
    business1 = search_df.loc[business_id1]
    business2 = search_df.loc[business_id2]
    business3 = search_df.loc[business_id3]

    business_1_description = f"{business1.english_name} is located {business1.airbnb_range} miles from your airbnb.\n\nThe positive reviews say: {business1.positive_summary}\n\nThe negative reviews say: {business1.negative_summary}"
    try:
        b1_img = Image.open(f"./Data/photos/{photos_df.loc[photos_df['business_id'] == business_id1]['photo_id'].values[0]}.jpg")
    except:
        b1_img = no_photos_img

    business_2_description = f"{business2.english_name} is located {business2.airbnb_range} miles from your airbnb.\n\nThe positive reviews say: {business2.positive_summary}\n\nThe negative reviews say: {business2.negative_summary}"
    try:
        b2_img = Image.open(f"./Data/photos/{photos_df.loc[photos_df['business_id'] == business_id2]['photo_id'].values[0]}.jpg")
    except:
        b2_img = no_photos_img

    business_3_description = f"{business3.english_name} is located {business3.airbnb_range} miles from your airbnb.\n\nThe positive reviews say: {business3.positive_summary}\n\nThe negative reviews say: {business3.negative_summary}"
    try:
        b3_img = Image.open(f"./Data/photos/{photos_df.loc[photos_df['business_id'] == business_id3]['photo_id'].values[0]}.jpg")
    except:
        b3_img = no_photos_img
    return business_1_description, b1_img, business_2_description, b2_img, business_3_description, b3_img

def update_options(choice):
    if choice == "Specific Food":
        return gr.Dropdown(choices=valid_types)
    elif choice == "Category":
        return gr.Dropdown(choices=list(list_of_cats.keys()))
    return []  

reset_user()

theme1 = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="red",
    radius_size="lg",
)

with gr.Blocks(title='Restaurant Recommendations') as Restaurants:
    with gr.Row():
        with gr.Row():
            choice = gr.Radio(["Specific Food", "Category"], label="What would you like to do?")
            option = gr.Dropdown(['Select Specific Food or Category'],label="Choose an option", value='Select Specific Food or Category', scale=2)
        with gr.Row():
            price_range = gr.CheckboxGroup(['$','$$','$$$','$$$$'], label="Price Range", info="What Price Ranges are you feeling?")
            indoor_outdoor = gr.Radio(['Indoor Seating', 'Outdoor Seating'], label='Indoor or Outdoor seating?',)
    with gr.Row():
        drinks = gr.CheckboxGroup(["Doesn't Matter","Beer and Wine","Full Bar","None"], label="Alcohol Available?", info="Select what type of drinks you would like available.")
        distance_slider = gr.Slider(value=5, minimum=0.1, maximum=30, label='Max distance from your AirBNB', interactive=True)
    with gr.Row():   
        submit_btn = gr.Button("Submit")
        reset_button = gr.Button('Reset User Preferences')

    with gr.Row():
        output1 = gr.Textbox(scale=3, visible=True, show_label=False, interactive=False)
        photo1 = gr.Image(scale=1, visible=True,show_label=False)
    with gr.Row():
        output2 = gr.Textbox(scale=3, visible=True, show_label=False, interactive=False)
        photo2 = gr.Image(scale=1, visible=True,show_label=False)
    with gr.Row():
        output3 = gr.Textbox(scale=3, visible=True, show_label=False, interactive=False)
        photo3 = gr.Image(scale=1, visible=True,show_label=False)

    choice.input(update_options, inputs=choice, outputs=option)
    submit_btn.click(fn=calculate_best_restaurant, inputs=[choice, option, price_range, indoor_outdoor, drinks, distance_slider], outputs=[output1, photo1, output2, photo2, output3,photo3])
    reset_button.click(fn=reset_user)

with gr.Blocks(theme=theme1) as site_seeing:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            city = gr.Textbox(label="City")
            days = gr.Number(label="Days", step=1)
            activities_per_day = gr.Number(label="Activities Per Day", step=1)
        with gr.Column(scale=2, min_width=500):
            top_priority = gr.Radio(label="Top Priority", choices=["Music", "Food", "Museums", "Sports"])
            other_preferences = gr.CheckboxGroup(label="Other Preferences", choices=["Adventure", "Food", "Michelin Star Meals", "Museums", "Nature", "Sports", "Wine Tasting"])
            special_requests = gr.Textbox(label="Special Requests")
    with gr.Row():
        submit = gr.Button("Generate Itinerary")
        
    
    itinerary_output = gr.Textbox(label="Recommended Itinerary", interactive=False)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            click_me = gr.Radio(['Yes','No'], label="Do you like this itinerary?")
        with gr.Column(scale=2, min_width=500):
            user_feedback = gr.Textbox(label="What didn't you like about it?", interactive=True)
    with gr.Row():
        submit2 = gr.Button("Generate New Itinerary")
   
    # Define callbacks
    submit.click(fn=generate_itinerary, inputs=[city, days, activities_per_day, top_priority, other_preferences, special_requests], outputs=itinerary_output)
    click_me.change(update_visibility, inputs=[click_me], outputs=[user_feedback, submit2])
    submit2.click(fn=feedback_itinerary, inputs=[user_feedback], outputs=itinerary_output)


interface = gr.TabbedInterface([Restaurants, site_seeing], ["Restaurants","Sightseeing"], theme=theme1)


if __name__ ==  '__main__':
    interface.launch(debug=True)