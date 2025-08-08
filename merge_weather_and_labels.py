import os 
import pandas as pd

weather_folder = "weather_data"
fire_label_file = "canada_fire_grid.csv"
output_file = "labeled_weather_data.csv"

# Load fire labels(columns: lat, lon, historical_fire)
fire_df = pd.read_csv(fire_label_file)

#Load and combine all weather data files
all_weather = []
for file_name in os.listdir(weather_folder):
  if file_name.endswith(".csv"):
    file_path = os.path.join(weather_folder, file_name)
    weather_df = pd.read_csv(file_path)

     #If no 'date' column, extract date from filename
    if "date" not in weather_df.columns:
      weather_df["date"] = file_name.replace(".csv", "")

    all_weather.append(weather_df)

#Combine all weather DataFrames into one
weather_all_df = pd.concat(all_weather, ignore_index=True) #Resets index after concatenation to avoid duplicates.

#Merge weather and fire data using lat/lon as keys
#'how="inner"' keeps only rows where lat/lon exist in BOTH datasets
#Automatically avoids duplicating lat/lon columns in the output
merged_df = pd.merge(
    weather_all_df,
    fire_df,
    on=["lat", "lon"],  #Merge key (shared columns)
    how="inner"         #Only keep matching rows
)

#NOTE:
#pd.concat: Combining similar datasets (e.g., multiple daily weather files)
#pd.merge: Merging related datasets (e.g., weather + fire labels)

#Save merged dataset 
merged_df.to_csv(output_file, index=False)
print(f"Labeled dataset saved as: {output_file}")
