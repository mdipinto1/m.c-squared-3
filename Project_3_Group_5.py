import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
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
from IPython.display import HTML 
import pandas as pd
from datetime import datetime, timedelta
import csv
from openpyxl import Workbook
import numpy as np
import gradio as gr
import re
import folium
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.agents import initialize_agent, load_tools
import openai
import datetime
import json
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer
from nltk.corpus import reuters
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.corpus import stopwords
import requests
import time

nltk.download('punkt')
nltk.download("reuters")
nltk.download('punkt')

load_dotenv('../my_keys.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"

################################################
################################################
################################################
################################################
################################################


current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%Y-%m-%d_%H-%M-%S")
dest_dir = './Sihong Zhou/'
file_name = f"Download{formatted_date}.csv"
file_nameUnique=f"BNBDailyAna.csv"
file_nameDP = f"dpDownload{formatted_date}.csv"
file_valid_stockDF=f"Valid{formatted_date}.csv"
file_dailyAna=f"dailyAna{formatted_date}.csv"
full_pathTest=f"cleanedAnaTEST.xlsx"
full_Rank=f"RankedDATA.xlsx"
cleanedAna=f"cleanedAna{formatted_date}.csv"

fileListing=pd.read_csv('../m.c-squared-3/Sihong Zhou/listings.csv')
fileListing.columns

file_to_modify=fileListing[['id', 'listing_url', 'last_scraped', 'neighborhood_overview',
       'host_name','host_response_time',
       'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
       'host_neighbourhood','neighbourhood_cleansed',
       'neighbourhood_group_cleansed',  
        'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds','amenities', 
       'price','latitude',
       'longitude', 
       'minimum_nights',
       'maximum_nights', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value', 
       ]]

file_to_modify.head()

def csv_to_excel(csv_file, excel_file):
    wb = Workbook()
    ws = wb.active

    with open(csv_file, 'r') as f:
        for row in csv.reader(f):
            ws.append(row)

    wb.save(excel_file)

csvfile=file_to_modify.to_csv(dest_dir+file_nameUnique)
PathExcel=dest_dir+full_pathTest

execelFile=file_to_modify.to_excel(PathExcel, sheet_name='MySheet', index=False)

df = pd.read_excel(PathExcel)

df_unique = df.drop_duplicates()

print(PathExcel)

final_path=f'{dest_dir}/UpdatedAna.xlsx'

df_unique.to_excel(final_path, sheet_name='Unique', index=True)

anaD=pd.read_excel(final_path,sheet_name='Unique')
anaD.head()

anaD.columns

CleanedData= pd.read_excel(final_path, usecols=['id','listing_url','host_response_rate', 'host_acceptance_rate','accommodates',
       'bathrooms', 'bedrooms', 'beds', 'price',
       'latitude', 'longitude','maximum_nights', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value'])
CleanedData.fillna(0, inplace=True)
CleanedData.head()

filtered_df = CleanedData[(CleanedData['review_scores_rating'] > 3) & (CleanedData['review_scores_accuracy'] >3)& (CleanedData['review_scores_value'] >3)]



filtered_df=filtered_df.dropna(subset=['price'])

filtered_df.head()


print(filtered_df.describe())


filtered_df.columns


df_sorted_multi=filtered_df.copy()


df_sorted_multi['priceFormated'] = df_sorted_multi['price'].replace('[\$,]', '', regex=True)
df_sorted_multi['priceFormated'] = pd.to_numeric(df_sorted_multi['priceFormated'], errors='coerce').fillna(0)


df_sorted_multi['priceFormated'] = df_sorted_multi['priceFormated'].astype(int)

df_sorted_multi['price_accommodates_ratio']= (df_sorted_multi['priceFormated']/df_sorted_multi['accommodates'])
df_sorted_multi['price_beds_ratio']= (df_sorted_multi['priceFormated']/df_sorted_multi['beds'])
df_sorted_multi['price_accommodates_ratioRank']= (df_sorted_multi['priceFormated']/df_sorted_multi['accommodates']).rank(method='first', ascending=False)
df_sorted_multi['price_beds_ratioRank']= (df_sorted_multi['priceFormated']/df_sorted_multi['beds']).rank(method='first', ascending=False)


df_sorted_multi['priceFormated'].value_counts



data_matching_id = df_sorted_multi[df_sorted_multi['id'] == 52217673]
test=data_matching_id[['priceFormated','beds','accommodates','price_accommodates_ratioRank','price_beds_ratioRank','price_accommodates_ratio','price_beds_ratio']]
test


columns_to_convert1 = ['host_response_rate', 'host_acceptance_rate', 'review_scores_rating',
                      'review_scores_accuracy', 'review_scores_cleanliness',
                      'review_scores_checkin', 'review_scores_communication',
                      'review_scores_location', 'review_scores_value']

for column in columns_to_convert1:

    df_sorted_multi[column] = df_sorted_multi[column].astype(str).str.replace('%', '')

    
    df_sorted_multi[column] = pd.to_numeric(df_sorted_multi[column], errors='coerce')

    
    df_sorted_multi[column] = df_sorted_multi[column].rank(method='first', ascending=False)


df_sorted_multi['host_responseRank'] = df_sorted_multi['host_response_rate'].rank(method='first', ascending=False)
df_sorted_multi['host_acceptanceRank'] = df_sorted_multi['host_acceptance_rate'].rank(method='first', ascending=False)
df_sorted_multi['review_scores_ratingRank'] = df_sorted_multi['review_scores_rating'].rank(method='first', ascending=False)
df_sorted_multi['review_scores_accuracyRank'] = df_sorted_multi['review_scores_accuracy'].rank(method='first', ascending=False)
df_sorted_multi['review_scores_cleanlinessRank'] = df_sorted_multi['review_scores_cleanliness'].rank(method='first', ascending=False)
df_sorted_multi['review_scores_checkinRank'] = df_sorted_multi['review_scores_checkin'].rank(method='first', ascending=False)
df_sorted_multi['review_scores_communicationRank'] = df_sorted_multi['review_scores_communication'].rank(method='first', ascending=False)
df_sorted_multi['review_scores_locationRank'] = df_sorted_multi['review_scores_location'].rank(method='first', ascending=False)
df_sorted_multi['review_scores_valueRank'] = df_sorted_multi['review_scores_value'].rank(method='first', ascending=False)
df_sorted_multi.head()


columns_to_convert = [
    'price_accommodates_ratioRank', 'price_beds_ratioRank', 'host_responseRank',
    'host_acceptance_rate', 'review_scores_rating', 'review_scores_accuracy',
    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value'
]

for column in columns_to_convert:
    df_sorted_multi[column] = pd.to_numeric(df_sorted_multi[column], errors='coerce')


print(df_sorted_multi[columns_to_convert].isna().sum())




def normalize(series, is_inverted=False):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:  
        return pd.Series([0.5]*len(series)) if is_inverted else pd.Series([0.5]*len(series))
    if is_inverted:
        return 1 - (series - min_val) / (max_val - min_val)
    return (series - min_val) / (max_val - min_val)


df_sorted_multi['normalized_price_accommodates_ratio'] = normalize(df_sorted_multi['price_accommodates_ratioRank'], is_inverted=True)
df_sorted_multi['normalized_price_beds_ratio'] = normalize(df_sorted_multi['price_beds_ratioRank'], is_inverted=True)
df_sorted_multi['normalized_host_response'] = normalize(df_sorted_multi['host_responseRank'])
df_sorted_multi['normalized_host_acceptance_rate'] = normalize(df_sorted_multi['host_acceptance_rate'])
df_sorted_multi['normalized_review_scores_rating'] = normalize(df_sorted_multi['review_scores_rating'])
df_sorted_multi['normalized_review_scores_accuracy'] = normalize(df_sorted_multi['review_scores_accuracy'])
df_sorted_multi['normalized_review_scores_cleanliness'] = normalize(df_sorted_multi['review_scores_cleanliness'])
df_sorted_multi['normalized_review_scores_checkin'] = normalize(df_sorted_multi['review_scores_checkin'])
df_sorted_multi['normalized_review_scores_location'] = normalize(df_sorted_multi['review_scores_location'])
df_sorted_multi['normalized_review_scores_value'] = normalize(df_sorted_multi['review_scores_value'])


df_sorted_multi['score'] = (
    df_sorted_multi['normalized_price_accommodates_ratio'] +
    df_sorted_multi['normalized_price_beds_ratio'] +


    df_sorted_multi['normalized_review_scores_rating']+
    df_sorted_multi['normalized_review_scores_accuracy']+
    df_sorted_multi['normalized_review_scores_cleanliness']+
    df_sorted_multi['normalized_review_scores_checkin']


)


df_sorted_multi.sort_values(by='score', ascending=False, inplace=True)


print(df_sorted_multi[['listing_url','latitude',
       'longitude', 'score']])


print(df_sorted_multi.dtypes)


rank_columns = columns_to_convert  
df_sorted_multi['AverageRank'] = df_sorted_multi[rank_columns].mean(axis=1)
overall_sorting=df_sorted_multi.sort_values(by='AverageRank')
best_option = df_sorted_multi.sort_values(by='AverageRank').iloc[0:2]

best_option


full_RankTest=dest_dir + full_Rank
overall_sorting.to_excel(full_RankTest, sheet_name='Unique', index=True)


df = pd.read_excel(full_RankTest)
value = df.iloc[1:2]
value



filter_df = overall_sorting[overall_sorting['accommodates'] == 3]
finaldf=filter_df[['listing_url','latitude', 'longitude']]
finaldf.head(5)


overall_sorting.columns


df_filtered_price = df_sorted_multi[df_sorted_multi['priceFormated'] > 0]
df_filtered_price.head(1)


df_filtered_price['priceFormated'].value_counts


print(f"There are {len(df_filtered_price['priceFormated'])} prices in airbnb Nashiville Tennense, and the range are from {df_filtered_price['priceFormated'].min()} to {df_filtered_price['priceFormated'].max()}, the middle price is {df_filtered_price['priceFormated'].mean()}")





df = pd.read_excel(full_RankTest)




parameters = ['price_accommodates_ratio', 'price_beds_ratio',
    'host_responseRank', 'host_acceptanceRank', 'review_scores_ratingRank',
    'review_scores_accuracyRank', 'review_scores_cleanlinessRank',
    'review_scores_checkinRank', 'review_scores_communicationRank',
    'review_scores_locationRank', 'review_scores_valueRank'
]


active_parameters = []
normalization_list=[]


def add_parameter(param_name, weight):
    """
    Adds a new parameter with its adjusted weight to the list if it's not already present.
    """
    if param_name and not any(p['name'] == param_name for p in active_parameters):
        adjusted_weight = calculate_adjusted_weight(weight)
        active_parameters.append({'name': param_name, 'weight': adjusted_weight})
        for param in active_parameters:

            
            if 'name' in param and param['name'] == 'price_accommodates_ratioRank':
                
                normalization_list.append('normalized_price_accommodates_ratio')
            if 'name' in param and param['name'] == 'price_beds_ratioRank':
                
                normalization_list.append('normalized_price_beds_ratio')
            if 'name' in param and param['name'] == 'host_responseRank':
                
                normalization_list.append('normalized_host_response')
            if 'name' in param and param['name'] == 'host_acceptanceRank':
                
                normalization_list.append('normalized_host_acceptance_rate')
            if 'name' in param and param['name'] == 'review_scores_rating':
                
                normalization_list.append('normalized_review_scores_rating')
            if 'name' in param and param['name'] == 'review_scores_accuracy':
                
                normalization_list.append('normalized_review_scores_accuracy')
            if 'name' in param and param['name'] == 'review_scores_cleanliness':
                
                normalization_list.append('normalized_review_scores_cleanliness')
            if 'name' in param and param['name'] == 'review_scores_checkin':
                
                normalization_list.append('normalized_review_scores_checkin')
            if 'name' in param and param['name'] == 'review_scores_location':
                
                normalization_list.append('normalized_review_scores_location')
            if 'name' in param and param['name'] == 'review_scores_value':
                
                normalization_list.append('normalized_review_scores_value') 
    return show_parameters()

def remove_parameter(param_name):
    """
    Removes a parameter from the list and recalculates weights for all remaining parameters.
    """
    global active_parameters
    active_parameters = [p for p in active_parameters if p['name'] != param_name]
    for param in active_parameters:
        
        original_slider_value = sum(p['weight'] for p in active_parameters) / len(active_parameters) * 5
        param['weight'] = calculate_adjusted_weight(original_slider_value)
    return show_parameters()

def calculate_adjusted_weight(slider_value):
    """
    Calculate the adjusted weight based on the slider value and the number of active parameters.
    Avoids division by zero by checking if the active_parameters list is empty.
    """
    if len(active_parameters) == 0:
        return slider_value / 5
    return (slider_value / 5) * (1 / len(active_parameters))

def show_parameters():
    """
    Returns a formatted string displaying the active parameters and their relative weights.
    """
    if not active_parameters:
        return "No parameters added."
    total_weight = sum(p['weight'] for p in active_parameters)
    result = [f"{param['name']} - Weight: {param['weight']} ({(param['weight'] / total_weight) * 100:.2f}%)" for param in active_parameters]
    return "\n".join(result)

parameter_names = [param['name'] for param in active_parameters]

def filter_prices(min_price, max_price):

    if min_value > max_value:
        min_value, max_value = max_value, min_value  
    if min_value<df_filtered_price['priceFormated'].min():
        print("Error please add a greater number")
    if min_value>df_filtered_price['priceFormated'].max():
        print("Error please add a lesser number")
    
    df_filtered_price = df_filtered_price[(df_filtered_price['priceFormated'] >= min_price) & (df_filtered_price['priceFormated'] <= max_price)]
    return df_filtered_price
    


def process_data(min_price, max_price):
   
    if not parameters:
        return "No criteria selected for ranking."
    
        
    ranking_present = any(param in df_filtered_price.columns for param in parameter_names)
    pricing_present = 'price_accommodates_ratio' in df_filtered_price.columns or 'price_beds_ratio' in df_sorted_multi.columns

    
    if ranking_present:
        
        existing_ranking_params = [param for param in parameter_names if param in active_parameters]
        df_filtered_price['AverageRank'] = df_filtered_price[existing_ranking_params].mean(axis=1)
        
    
    if pricing_present:
        
        primary_pricing_param = 'price_accommodates_ratio' if 'price_accommodates_ratio' in active_parameters else 'price_beds_ratio'
        df_filtered_price.sort_values(by=primary_pricing_param, ascending=True, inplace=True)
    else:
        df_filtered_price.sort_values(by='AverageRank', ascending=False, inplace=True)
    
    
    overall_sorting = df_filtered_price[(df_filtered_price['priceFormated'] >= min_price) & (df_filtered_price['priceFormated'] <= max_price)]
    
    best_option = overall_sorting.head(5)  
    full_RankTest=dest_dir + full_Rank
    overall_sorting.to_excel(full_RankTest, sheet_name='Unique', index=True)
    return best_option[['id', 
       'accommodates', 'bathrooms', 'beds', 'price','price_accommodates_ratio','price_accommodates_ratioRank', 'score', 'AverageRank']].to_html()  

   
        
def calculate_combined_score_forSpecifyDataframe(df, normalized_columns):

    
    missing_cols = [col for col in normalized_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {', '.join(missing_cols)}")

    
    df['score'] = df[normalized_columns].sum(axis=1)
    return df

def calculate_combined_score(normalized_columns):
    
    global df_filtered_price  

    
    missing_cols = [col for col in normalized_columns if col not in df_filtered_price.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {', '.join(missing_cols)}")

    
    df_filtered_price['score'] = df_filtered_price[normalized_columns].sum(axis=1)
    best_suggestion=df_filtered_price.head(5)
    return best_suggestion
"""def make_clickable(url):
    return f'<a href="{url}">{url}</a>'


df['listing_url'] = df['listing_url'].apply(make_clickable)


pd.set_option('display.html.use_mathjax', False)  
HTML(df.to_html(escape=False))
"""
locList=[]
maps_html = []
latitude=[]
longitude=[]
global filter_df_answer_questions
filter_df_answer_questions = pd.DataFrame()

def answer_questions(accommodates, price):
    global filter_df_answer_questions
    filter_df_answer_questions =df_filtered_price[(df_filtered_price['accommodates'] == accommodates) &(df['priceFormated'] > (price - 20))&(df['priceFormated'] < (price + 20))]
        
            
    try:
        for index, row in filter_df.iterrows():
            latitude.append(row['latitude'])
            longitude.append(row['longitude'])

    except IndexError:
        print(f"sorry its error")
    
        combined_html = "".join(maps_html) 
    try:
        row = filter_df_answer_questions[['listing_url','latitude', 'longitude']].head(10)
    except IndexError:
        row

    return row



def extract_id_from_string(s):
    
    match = re.search(r'/rooms/(\d+)', s)
    if match:
        return int(match.group(1))  
    else:
        return None  
def extract_and_convert_to_int(input_string):
    
    numeric_part = re.sub(r"[^\d]", "", input_string)
    return int(numeric_part)

def dataframe_to_list(filter_df_answer_questions):
    search_string = r'/(\d+)[^\d]*$'
    return [
        f"ID {re.search(search_string, row['listing_url']).group(1)} - Latitude: {row['latitude']}, Longitude: {row['longitude']}"
        for index, row in df.iterrows()
        if re.search(search_string, row['listing_url'])  
    ]

selected_items = []
"""def extract_and_convert_to_int(input_string):
    
    numeric_part = input_string.split()[1]
    return int(numeric_part)
"""
def extract_and_convert_to_int_regex(input_string):
    
    match = re.search(r'\d+', input_string)
    if match:
        return int(match.group(0))



"""def handle_selection(selected_option):

    selected_id_str = selected_option.split(" - ")[0]
    selected_id = extract_and_convert_to_int(selected_id_str)
    filtered_row = df[df['listing_url'].str.contains(str(selected_id))].iloc[0]    
    selected_items.append(filtered_row.to_dict())  
    selectedLan.append(filtered_row['latitude'])
    selectedLon.append(filtered_row['longitude'])
    return f"Selected: {filtered_row['listing_url']} with ID {selected_id} "
"""
def handle_selection(selected_option):
    reset_selections()
    global selectedLan, selectedLon  

    selected_id_str = selected_option.split(" - ")[0]
    selected_id = extract_and_convert_to_int(selected_id_str)

    filtered_row = df[df['listing_url'].str.contains(str(selected_id))].iloc[0]

    
    if filtered_row['latitude'] in selectedLan or filtered_row['longitude'] in selectedLon:
        reset_selections()  


    
    selectedLan.append(filtered_row['latitude'])
    selectedLon.append(filtered_row['longitude'])
    return f"Selected: {filtered_row['listing_url']} with ID {selected_id}"



def reset_selections():
    global selectedLan, selectedLon
    selectedLan = []
    selectedLon = []



def create_map(latitudes, longitudes):
    
    if latitudes and longitudes:
        central_lat = latitudes[0]
        central_lon = longitudes[0]
    else:
        central_lat = 0  
        central_lon = 0
    folium_map = folium.Map(location=[central_lat, central_lon], zoom_start=5)

    
    for lat, lon in zip(latitudes, longitudes):
        folium.Marker([lat, lon], tooltip='Click for info', popup=f'Coordinates: {lat}, {lon}').add_to(folium_map)

    
    return folium_map._repr_html_()

def generate_maps(latitudes, longitudes):
    maps_html = []
    for lat, lon in zip(latitudes, longitudes):
        map_html = create_map(lat, lon)
        maps_html.append(map_html)
    return maps_html

reset_selections()


with gr.Blocks() as airbnb_interface:
    with gr.Row():
        param_dropdown = gr.Dropdown(choices=parameters, label="Select Parameter")
        weight_slider = gr.Slider(minimum=1, maximum=10, step=1, value=0, label="Set Weight")
        add_btn = gr.Button("Add Parameter")
        remove_btn = gr.Button("Remove Parameter")
    
    output_text = gr.Textbox(label="Parameters and Weights", lines=10)
    
    # Setup event handlers
    add_btn.click(fn=add_parameter, inputs=[param_dropdown, weight_slider], outputs=output_text)
    remove_btn.click(fn=remove_parameter, inputs=param_dropdown, outputs=output_text)
    
    gr.Markdown("## Step 2: Process Data Based on Selected Criteria")
    with gr.Row():
        min_slider = gr.Slider(minimum=10, maximum=5186, step=1, value=10, label="Minimum Value")
        max_slider = gr.Slider(minimum=10, maximum=5186, step=1, value=5186, label="Maximum Value")
    process_btn = gr.Button("Process Data")
    result_output = gr.HTML(label="Best Options Based on Average Rank")


    process_btn.click(fn=process_data, inputs=[min_slider,max_slider], outputs=result_output)

    gr.Markdown("## Step 3: Find Listings")
    accommodates_input = gr.Number(label="Accommodates")
    price_input = gr.Number(label="Price")
    find_btn = gr.Button("Find Listings")
    listings_output = gr.Textbox(label="Listings")
    find_btn.click(fn=answer_questions, inputs=[accommodates_input, price_input], outputs=listings_output)  

    gr.Markdown("## Step 4: Select from the Suggested List")

    dropdown = gr.Dropdown(choices=dataframe_to_list(df))
    output_text = gr.Textbox()
    dropdown.change(handle_selection, inputs=dropdown, outputs=output_text)

print(f"{selectedLan},{selectedLon}")





latitudeMap = 36.15087
longitudeMap = -86.83155

map_html = create_map([latitudeMap], [longitudeMap])

def show_map():
    combined_html = "".join(map_html)  
    return combined_html

def update_lat_long():
    global latitudeMap
    global longitudeMap
    latitudeMap = selectedLan[0]
    longitudeMap = selectedLon[0]

def get_city_name(latitude, longitude, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'results' in data and data['results']:
        return data['results'][0]['formatted_address']
    else:
        return "Unknown"
api_key = os.getenv('WEATHER_API_KEY')
google_api = os.getenv('GOOGLE_API')
address = get_city_name(latitudeMap, longitudeMap, google_api)
print(address)




def get_Exact_city_name(address):
    
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={google_api}"
    
    
    response = requests.get(url)
    data = response.json()
    
    
    if 'results' in data and len(data['results']) > 0:
        
        for component in data['results'][0]['address_components']:
            if 'locality' in component['types']:
                return component['long_name']
    
    return "City not found"


city_name = get_Exact_city_name(address)
print("City Name:", city_name)

def update_all_weather_stuff():
    update_lat_long()
    print(latitudeMap)
    print(longitudeMap)
    global address 
    address = get_city_name(latitudeMap, longitudeMap, google_api)
    print(address)
    global city_name 
    city_name = get_Exact_city_name(address)
    print(city_name)
    global map_html
    map_html = create_map(latitudeMap, longitudeMap)

def get_weather(city_name, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    df = pd.json_normalize(data)
    return data




weather_dict={}
output={}
def final_weatherInfo():
    weather_data = get_weather(city_name, api_key)
    print(weather_data)
    
    timestamp = weather_data.get('dt', None)
    if timestamp:
        date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    else:
        date = "No date available"

    weather_info = weather_data['weather']
    temperature = weather_data['main']
    visibility = weather_data['visibility']
    wind = weather_data['wind']

    
    global weather_dict
    weather_dict = {
        'Date': date,
        'Location': city_name,
        'Max Temperature': f"{temperature['temp_max']:.1f}°C",
        'Min Temperature': f"{temperature['temp_min']:.1f}°C",
        'Weather': weather_info[0]['main'],
        'Humidity': f"{temperature['humidity']}%",
        'Visibility': f"{visibility/1000:.1f} km",  
        'Wind Speed': f"{wind['speed']} m/s"
    }

    def format_weather_output(weather_dict):
        global output
        output = (
            f"Date: {weather_dict['Date']}\n"
            f"Location: {weather_dict['Location']}\n"
            f"Max Temperature: {weather_dict['Max Temperature']}\n"
            f"Min Temperature: {weather_dict['Min Temperature']}\n"
            f"Weather: {weather_dict['Weather']}\n"
            f"Humidity: {weather_dict['Humidity']}\n"
            f"Visibility: {weather_dict['Visibility']}\n"
            f"Wind Speed: {weather_dict['Wind Speed']}\n"
        )
    format_weather_output(weather_dict)
    return output




selectedLon


selectedLan




country_code='US'
def load_city_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        cities = json.load(file)
    return cities

def find_city_id(city_name,country_code, city_list):
    """
    Search for a city ID by city name and country code to avoid duplicates in different countries.
    """
    for city in city_list:
        if city['name'].lower() == city_name.lower() and city['country'].lower() == country_code.lower():
            return city['id']
    return None


filepath = './Sihong Zhou/city.list.json'  
city_list = load_city_list(filepath)
city_id = find_city_id(city_name,country_code, city_list)  
print("City ID:", city_id)


def fetch_weather_forecast(api_key):
  
    url = f"http://api.openweathermap.org/data/2.5/forecast?id={city_id}&appid={api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Failed to retrieve data:", response.status_code)
        return None
weekly_weather=[]




def display_forecast(data):
    if data:
        grouped_by_date = defaultdict(list)

        
        for entry in data['list']:
            time = entry['dt_txt']
            date = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S').date()  
            
            temperature = f"{entry['main']['temp'] - 273.15:.2f}°C"
            description = entry['weather'][0]['description']
            formatted_forecast = f"{time:<20} {temperature:>12} {description:<20}"
            grouped_by_date[date].append(formatted_forecast)

        
        weekly_weather = []
        for date, forecasts in sorted(grouped_by_date.items()):  
            header = f"Date: {date}\n{'Time':<20} {'Temperature':>12} {'Description':<20}"
            weekly_weather.append(header)
            weekly_weather.append('-' * len(header))
            for forecast in forecasts:
                weekly_weather.append(forecast)

        return "\n".join(weekly_weather)
    
    
def get_weekly_weather():
    forecast_data = fetch_weather_forecast(api_key)
    
    return display_forecast(forecast_data)


"""display for dataframe
import pandas as pd

def display_forecast(data):
    if data:
        
        times = []
        temperatures = []
        descriptions = []
        
        for entry in data['list']:
            times.append(entry['dt_txt'])
            temperatures.append(f"{entry['main']['temp'] - 273.15:.2f}°C")  
            descriptions.append(entry['weather'][0]['description'])
        
        
        df = pd.DataFrame({
            'Time': times,
            'Temperature': temperatures,
            'Description': descriptions
        })
        
        
        return df.to_string(index=False)
"""


def Activity_Planner():
    
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.3)
    
    tools = load_tools (["openweathermap-api"], openweathermap_api_key=api_key, llm=llm)
    agent = initialize_agent(tools, 
                            agent="chat-zero-shot-react-description",
                            handle_parsing_errors = True,
                            max_iterations=10,
                            
                            llm=llm)
    
    location = city_name
    
    query = {"input": f"""
            Please suggest an activity for a tourist today in {location}. 
            The activity should be appropriate to the current weather.
            Try to name specific places whenever possible.
            """}
    result = agent.invoke(query)
    return result["output"]


with gr.Blocks() as weather_interface:
    import_airbnb_button_weather = gr.Button('Import Airbnb')
    gr.Markdown("#Step 5 map the searched list")

    map_output = gr.HTML(show_map())

    gr.Markdown("## Step 6: Get the Weather")
    
    weather_btn = gr.Button("Weather Button")
    weather_output = gr.Textbox(label="weather")
    weather_btn.click(fn=final_weatherInfo, inputs=[], outputs=weather_output) 

    gr.Markdown("## Step 7: Get the weekly Weather")
    
    week_weather_btn = gr.Button("Week Weather Button")
    week_weather_output = gr.Textbox(label="weekly weather")
    week_weather_btn.click(fn=get_weekly_weather, inputs=[], outputs=week_weather_output) 
    import_airbnb_button_weather.click(fn=update_all_weather_stuff)
# app.launch('share=True')




# stop_words = set(stopwords.words('english'))



# sentence_1 = "I want to travel in June."
# sentence_2 = "Should I prepare raincoat to Nashville?"
# sentence_3 = "What is the weather look like in mid of Jun and what activity do you recommend if I go to Nashville?"





# pattern = r'[^a-zA-Z\s ]'


# tokens = []


# sentence_1_cleaned = re.sub(pattern, '', sentence_1)
# sentence_1_tokens = nltk.word_tokenize(sentence_1_cleaned.lower())
# tokens.append(sentence_1_tokens)


# sentence_2_cleaned = re.sub(pattern, '', sentence_2)
# sentence_2_tokens = nltk.word_tokenize(sentence_2_cleaned.lower())
# tokens.append(sentence_2_tokens)


# sentence_3_cleaned = re.sub(pattern, '', sentence_3)
# sentence_3_tokens = nltk.word_tokenize(sentence_3_cleaned.lower())
# tokens.append(sentence_3_tokens)


# tokens



# filtered_tokens = []
# for token in tokens:
#     filtered_token = [word for word in token if not word in stop_words]
#     filtered_tokens.append(filtered_token)
    

# filtered_tokens



# bag_of_words = {}
# for i in range(len(filtered_tokens)):
#     for word in filtered_tokens[i]:
#         if word not in bag_of_words:
#             bag_of_words[word] = 0
#         bag_of_words[word] += 1


# print(bag_of_words)




# vectorizer = CountVectorizer(stop_words='english')


# bag_of_words = vectorizer.fit_transform([sentence_1,sentence_2, sentence_3])


# print(bag_of_words.toarray())


# bow_df = pd.DataFrame(bag_of_words.toarray(),columns=vectorizer.get_feature_names_out())
# bow_df


# print(bow_df.columns.to_list())


# occurrence = bow_df.sum(axis=0)
# print(occurrence)




# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# text = "I want to travel to Nashville in Jun." 


# subwords = tokenizer.tokenize(text)
# subwords



# print(reuters.categories())



# reuters.fileids(categories = 'cpi')[0]


# article = reuters.raw('test/14891')
# print(article)



# spacy.cli.download("en_core_web_sm")
# nlp = spacy.load("en_core_web_sm")



# sent_tokenize(article)

# sent = sent_tokenize(article)[0]
# print(sent)


# word_tokenize(sent)



# nlp = spacy.load("en_core_web_sm")

# spacy_sent = nlp(sent)
# [token.text for token in spacy_sent]

# sentence_subwords = tokenizer.tokenize(sent)
# sentence_subwords





# df


# df_sorted_multi.columns


# Clustering_data=df_sorted_multi[['id','host_response_rate',
#        'host_acceptance_rate','review_scores_rating', 'review_scores_accuracy',
#        'review_scores_cleanliness', 'review_scores_checkin',
#        'review_scores_communication', 'review_scores_location',
#        'review_scores_value','price_accommodates_ratio',
#        'price_beds_ratio',]]


# Clustering_data.sort_values(by='review_scores_rating')




# Clustering_data.plot.scatter(x="review_scores_rating",
#                                 y="review_scores_cleanliness")
       






# model = KMeans(n_clusters=2, n_init='auto', random_state=1)


# model


# ClusterTest=Clustering_data[["review_scores_rating","review_scores_cleanliness"]]



# model.fit(ClusterTest)



# customer_ratings = model.predict(ClusterTest)


# print(customer_ratings)



# service_rating_predictions_df = ClusterTest.copy()


# service_rating_predictions_df['customer rating'] = customer_ratings


# service_rating_predictions_df.head()



# service_rating_predictions_df.plot.scatter(
#     x="review_scores_rating", 
#     y="review_scores_cleanliness",
#     c="customer rating", 
#     colormap='rainbow')





# fileListing.columns


# KMEAN_Data=fileListing[['property_type','room_type', 'accommodates', 'bathrooms', 'bedrooms', 'price','instant_bookable']]
# KMEAN_Data.drop_duplicates

# KMEAN_updated=KMEAN_Data.fillna('0')
# KMEAN_updated


# KMEAN_updated.info()



# KMEAN_updated['instant_bookable']


# KMEAN_Data['room_type'].value_counts()


# KMEAN_Data['property_type'].value_counts()



# def encodeMethod(instant_bookable):

#     if instant_bookable == "f":
#         return 1
#     else:
#         return 2
# def encodeRoom_type(room_type):

#     if room_type == "Entire home/apt":
#         return 1
#     elif room_type == "Private room":
#         return 2
#     elif room_type == "Hotel room":
#         return 3
#     elif room_type == "Shared room":
#         return 4
# def encodeProperty_type(property_type):

#     if property_type == "Entire home":
#         return 1
#     elif property_type == "Entire rental unit":
#         return 2
#     elif property_type == "Entire condo":
#         return 3
#     elif property_type == "Entire townhouse ":
#         return 4
#     elif property_type == "Private room in home ":
#         return 5
#     elif property_type == "Entire guest suite":
#         return 6
#     elif property_type == "Room in hotel  ":
#         return 7
#     elif property_type == "Entire guesthouse ":
#         return 8
#     elif property_type == "Entire loft ":
#         return 9
#     elif property_type == "Entire serviced apartment ":
#         return 10
#     elif property_type == "Room in boutique hotel":
#         return 11
#     elif property_type == "Private room in resort  ":
#         return 12
#     elif property_type == "Entire bungalow":
#         return 13
#     elif property_type == "Room in aparthotel":
#         return 14
#     else: 
#         return 100



# KMEAN_updated['instant_bookable'] = KMEAN_updated['instant_bookable'].apply(encodeMethod)
# KMEAN_updated['room_type'] = KMEAN_updated['room_type'].apply(encodeRoom_type)
# KMEAN_updated['property_type'] = KMEAN_updated['property_type'].apply(encodeProperty_type)



# KMEAN_updated['room_type'].value_counts()


# KMEAN_updated['room_type'] = KMEAN_updated['room_type'].replace([None, 'None'], 0)



# KMEAN_updated['price'] = pd.to_numeric(df['price'].str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0).astype(int)



# KMEAN_updated.head(2)



# model_k2 = KMeans(n_clusters=2, n_init='auto')


# KMEAN_updated.fillna(0, inplace=True)



# model_k2.fit(KMEAN_updated)



# customer_segments_k2 = model_k2.predict(KMEAN_updated)


# print(customer_segments_k2)



# model_k3 = KMeans(n_clusters=3, n_init='auto')



# model_k3.fit(KMEAN_updated)



# customer_segments_k3 = model_k3.predict(KMEAN_updated)


# print(customer_segments_k3)



# instant_bookable_predict = KMEAN_updated.copy()

# instant_bookable_predict["Customer Segment (k=2)"] = customer_segments_k2



# instant_bookable_predict["Customer Segment (k=3)"] = customer_segments_k3


# instant_bookable_predict.head()



# instant_bookable_predict.plot.scatter(
#     x="instant_bookable", 
#     y="price", 
#     c="Customer Segment (k=2)",
#     title = "Scatter Plot by room - k=2",
#     colormap='winter'
# )



# instant_bookable_predict.plot.scatter(
#     x="instant_bookable", 
#     y="accommodates", 
#     c="Customer Segment (k=3)",
#     title = "Scatter Plot by room type - k=3",
#     colormap='winter'
# )



# inertia = []


# k = list(range(1, 11))


# for i in k:
#     model = KMeans(n_clusters=i, n_init='auto', random_state=1)
#     model.fit(KMEAN_updated)
#     inertia.append(model.inertia_)



# elbow_data = {
#     "k": k,
#     "inertia": inertia
# }


# df_elbow = pd.DataFrame(elbow_data)


# df_elbow



# df_elbow.plot.line(x="k",
#                    y="inertia",
#                    title="Elbow Curve",
#                    xticks=k)



# k = elbow_data["k"]
# inertia = elbow_data["inertia"]
# for i in range(1, len(k)):
#     percentage_decrease = (inertia[i-1] - inertia[i]) / inertia[i-1] * 100
#     print(f"Percentage decrease from k={k[i-1]} to k={k[i]}: {percentage_decrease:.2f}%")



# model = KMeans(n_clusters=4, n_init='auto', random_state=1)


# model.fit(KMEAN_updated)


# k_4 = model.predict(KMEAN_updated)


# instantBookable_predictions_df = KMEAN_updated.copy()


# instantBookable_predictions_df['book_segment'] = k_4


# instantBookable_predictions_df


# instantBookable_predictions_df.plot.scatter(
#     x="instant_bookable", 
#     y="price", 
#     c="book_segment",
#     title = "Scatter Plot by instant bookable Segment - k=4",
#     colormap='viridis'
# )








# KMEAN_updated.columns
# cleanK = KMEAN_updated[KMEAN_updated['room_type'] != 0]



# RoomTypeEvaluation=cleanK[['room_type','price']]
# RoomTypeEvaluation.head()


# X = RoomTypeEvaluation["price"].values.reshape(-1, 1)


# X[:5]


# X.shape



# y = RoomTypeEvaluation["price"]



# model = LinearRegression()

# model.fit(X, y)



# predicted_y_values = model.predict(X)


# RoomTypeEvaluation_predicted = RoomTypeEvaluation.copy()


# RoomTypeEvaluation_predicted["price"] = predicted_y_values


# RoomTypeEvaluation_predicted.head()






# score = round(model.score(X, y, sample_weight=None),5)
# r2 = round(r2_score(y, predicted_y_values),5)
# mse = round(mean_squared_error(y, predicted_y_values),4)
# rmse = round(np.sqrt(mse),4)


# # print(f"The score is {score}.")
# # print(f"The r2 is {r2}.")
# # print(f"The mean squared error is {mse}.")
# # print(f"The root mean squared error is {rmse}.")



################################################
################################################
################################################
################################################
################################################

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


################################################
################################################
################################################
################################################
################################################



business_df = pd.read_json('./Data/yelp_academic_dataset_business.json', lines=True)
business_df = business_df[(business_df['state'] == 'TN') & (business_df['is_open'] == 1)]
business_df['categories'] = business_df['categories'].fillna('')
business_df = business_df[business_df['categories'].str.contains('Restaurants')]

photos_df = pd.read_json('./Data/photos.json', lines=True)
photos_df = photos_df.loc[photos_df['business_id'].isin(business_df['business_id'])]

checkin_df = pd.read_json('./Data/yelp_academic_dataset_checkin.json', lines=True)
checkin_df = checkin_df.loc[checkin_df['business_id'].isin(business_df['business_id'])]

# useful_chunks = []

# for chunk in pd.read_json('./Data/yelp_academic_dataset_review.json', lines=True, chunksize=100000):
#     filtered_chunk = chunk.loc[chunk['business_id'].isin(business_df['business_id'])]
#     useful_chunks.append(filtered_chunk)

# reviews_df = pd.concat(useful_chunks)
# reviews_df.reset_index(drop=True, inplace=True)

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

def populate_range_df():#airbnb_lat, airbnb_long
    global selectedLan
    global selectedLon
    # print(f"selected lan is {selectedLan} |||| lon is {selectedLon}")
    # print(selectedLon)
    # print(selectedLan)
    # print(selectedLon[0])
    # print(selectedLan[0])
    airbnb_lat = selectedLan[0]
    airbnb_long = selectedLon[0]
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
    with gr.Row(variant='compact'):
            import_airbnb_button = gr.Button('Import BnB')
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
    import_airbnb_button.click(fn=populate_range_df)

with gr.Blocks(theme=theme1) as site_seeing:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            city = gr.Textbox(label="City", value='Nashville')
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


interface = gr.TabbedInterface([airbnb_interface, weather_interface, Restaurants, site_seeing], ["Airbnbs", "Weather", "Restaurants", "Sightseeing"], theme=theme1)


if __name__ ==  '__main__':
    interface.launch(debug=True)