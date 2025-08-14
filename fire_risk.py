import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

#Load the labeled data
df = pd.read_csv("labeled_weather_data.csv")

def adjust_features(df):
   #Create additional features that are relevant for fire risk prediction

   #Temperature based features
   df['temp_range'] = df['temp_max'] - df['temp_min']
   df['is_hot'] = (df['temperature'] > 25).astype(int)

   #Humidity based features
   df['is_dry'] = (df['humidity'] < 30).astype(int)
   df['humidity_temp_ratio'] = df['humidity'] / (df['temperature'] + 1) #Avoid division by 0

   #Wind based features (high wind = higher fire risk)
   df['is_windy'] = (df['wind_speed'] > 5).astype(int)
   df['wind_gust_filled'] = df['wind_gust'].fillna(df['wind_speed'])

   #Precipitation features (recent rain/snow = lower fire risk)
   df['total_precip'] = df['rain_1h_mm'] + df['snow_1h_mm'] 
   df['has_recent_precip'] = (df['total_precip'] > 0).astype(int)
   #np.where(condition, if_true, if_false)
   df['days_since_precip'] = np.where(df['total_precip'] > 0, 0, 1)

   #Pressure features
   df['is_high_pressure'] = (df['pressure'] > 1020).astype(int)

   #Composite fire danger index, higher values = higher fire risk
   df['fire_danger_index'] = (
      (df['temperature'] / 40) * 0.3 +
      ((100 - df['humidity']) / 100) * 0.3 +
       (df['wind_speed'] / 20) * 0.2 +
       ((1040 - df['pressure']) / 100) * 0.1 +
        (1 - df['has_recent_precip']) * 0.1)
   
   return df

df = adjust_features(df)

#Create location groups to prevent leakage
df['location_group'] = df['lat'].astype(str) + '_' + df['lon'].astype(str)

#Split preprocessing: only use training data statistics
def safe_preprocessing(df_train, df_test):
   """Preprocess without data leakage"""

   #Calculate statistics ONLY from training data
   numeric_cols = df_train.select_dtypes(include=[np.number]).columns
   train_medians = df_train[numeric_cols].median()

   #Apply training statistics to both sets
   df_train[numeric_cols] = df_train[numeric_cols].fillna(train_medians)
   df_test[numeric_cols] = df_test[numeric_cols].fillna(train_medians)

   #LabelEncoder converts categorical text data into numerical labels
   label_encoder = LabelEncoder()
   #Create a new column with encoded weather categories
   train_weather = df_train['weather_main'].fillna('Unknown')
   df_train['weather_main_encoded'] = label_encoder.fit_transform(train_weather)

   #Handle unseen categories in test data
   test_weather = df_test['weather_main'].fillna('Unknown')
   test_encoded = []
   #Encode each weather condition in test set
   for weather in test_weather:
       #Check if this weather type exists in the training set categories
      if weather in label_encoder.classes_:
         #If it exists, transform it using the same encoding as training,[0] extracts the encoded value from the numpy array
         test_encoded.append(label_encoder.transform([weather])[0])
      else:
           #For unseen categories, use the most frequent weather type from training set as fallback
           #train_weather.mode()[0] gets the single most common weather in training
          test_encoded.append(label_encoder.transform(train_weather.mode()[0])[0])
          
   #Assign encoded values back to test dataframe
   df_test['weather_main_encoded'] = test_encoded
   return df_train, df_test, label_encoder

features = [
    'temperature', 'humidity', 'wind_speed', 'pressure',
    'temp_min', 'temp_max', 'feels_like',
    'wind_gust_filled', 'clouds_pct', 'visibility_m',
    'rain_1h_mm', 'rain_3h_mm', 'snow_1h_mm', 'snow_3h_mm',
    
    #Engineered weather features
    'temp_range', 'is_hot', 'is_dry', 'humidity_temp_ratio',
    'is_windy', 'total_precip', 'has_recent_precip',
    'is_high_pressure', 'fire_danger_index',
    
    #Encoded categorical features
    'weather_main_encoded'
]

#Ensures same locations don't appear in both train and test
#test_size=0.2: 20% of locations will be allocated to test set
#n_splits=1: Only perform one split (train/test)
#random_state=42: Ensures reproducible splits across runs
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

#The split() method returns indices for train/test sets 
#First(df): The full dataset (not actually used, just for dimensions)
#y=df['historical_fire']: The target variable (used for stratified splitting if needed)
# groups=df['location_group']: Defines the grouping structure
# next() extracts the first (and only) split from the generator
train_idx, test_idx = next(splitter.split(df, df['historical_fire'], groups=df['location_group']))

#Create the actual train/test DataFrames, using .iloc to select rows by position index
#.copy() ensures we get new DataFrames rather than views
df_train = df.iloc[train_idx].copy()#Contains 80% of unique locations
df_test = df.iloc[test_idx].copy()  #Contains 20% of unique locations

#Call function which handles missing values using only training set statistics, encodes categorical variables using only training set categories, returns the DataFrames along with the fitted encoder
df_train, df_test, label_encoder = safe_preprocessing(df_train, df_test)

#Only includes the columns specified in the 'features' list
#This creates the input variables the model will learn from
X_train = df_train[features]  #Training features (80% of locations)
X_test = df_test[features]    #Test features (20% of NEW locations)

#These are what the model will try to predict
y_train = df_train['historical_fire'] 
y_test = df_test['historical_fire']    #ground truth for evaluation

#Data Flow:
#Raw Data → Train/Test Split → Safe Preprocessing → Feature/Target Separation → Model
model = RandomForestClassifier(
   n_estimators=200, #More trees for better performance
   max_depth=15,  #Control overfitting
   min_samples_split=10, #Control overfitting
   min_samples_leaf=5, #Control overfitting
   class_weight='balanced', #Handle imbalanced dataset
   random_state=42,
   n_jobs=-1  #Use all CPU cores
)
model.fit(X_train, y_train) #Train the model

#Predict fire risk probabilities for test set features (X_test)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] #Prob fire happens

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Model trained successfully!")
print(f"Accuracy: {accuracy:.3f} | ROC AUC: {roc_auc:.3f}")

#Save model components
joblib.dump(model, "model_components/fire_risk_model.pkl")
joblib.dump(label_encoder, "model_components/weather_encoder.pkl")
joblib.dump(features, "model_components/model_features.pkl")
