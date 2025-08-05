import requests 
import json
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import os

#load API key 
with open("config.json") as f:
  config = json.load(f) #Load JSON into a Python dictionary

API_KEY = config["openweather_api_key"]

cities = [
    "Vancouver", "Kelowna", "Kamloops", "Calgary", "Edmonton","Fort McMurray", "Saskatoon", "Regina", "Winnipeg", "Thunder Bay","Ottawa", "Toronto", "Sudbury", "Montreal", "Quebec City","Halifax", "Whitehorse", "Yellowknife", "Prince George", "Victoria"
]

calgary_tz = ZoneInfo("America/Edmonton")  
calgary_time = datetime.now(calgary_tz) 

#get current weather for a city
# Returns a dictionary with weather details or None if failed.
def get_weather(city):
  url= f"http://api.openweathermap.org/data/2.5/weather?q={city},CA&appid={API_KEY}&units=metric"

  try:
     #Add timeout=5 so if the server doesn't respond within 5 seconds, skip it
    response = requests.get(url, timeout=10)

    #If the request succeeded
    if response.status_code == 200:
      #API response is JSON, so we call .json() to convert it into a Python dictionary.
      data = response.json() 

      return {
        "city": city,
        "date": calgary_time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": data["main"]["temp"], # degrees C
        "humidity": data["main"]["humidity"], # humid %
        "wind_speed": data["wind"]["speed"] # m/s
      }
    else: 
     print(f"Failed to fetch data for {city}")
     return None
  except Exception as e:
    print(f"Error fetching {city}")
    return None
  
#Prepare output csv file
#keep adding to this file over time for historical data
output_file = "weather_history.csv"

#Create the file with a header if it doesn't exist
#Check if the output CSV file already exists in the filesystem
if not os.path.exists(output_file):
    # Open the file in write mode ('w') to create a new file
    # Using newline="" prevents extra blank lines between rows in the CSV 
    with open(output_file, mode="w", newline="") as f:
        # Create a DictWriter object that maps dictionaries to CSV rows
        # fieldnames parameter defines the column headers for our CSV
        writer = csv.DictWriter(f, fieldnames=["city", "date", "temperature", "humidity", "wind_speed"])
        
        # Write the header row (column names) to the newly created file
        writer.writeheader()

#collect weather data for all cities
weather_data = []
for city in cities: 
  result = get_weather(city)
  if result:
    weather_data.append(result) #Add the dictionary to the list

#Append today's data to the csv
# Open the existing file in append mode ('a') to add new data without overwriting
with open(output_file, mode="a", newline="") as f:
    # Create a DictWriter with the same fieldnames structure as before
    writer = csv.DictWriter(f, fieldnames=["city", "date", "temperature", "humidity", "wind_speed"])
    
    #Write multiple rows of weather data to the CSV file
    writer.writerows(weather_data)

print(f"✅ Saved weather data for {len(weather_data)} cities at  {calgary_time} (Calgary Time)")

