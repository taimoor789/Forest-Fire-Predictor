import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib


#Load the labeled data
df = pd.read_csv("labeled_weather_data.csv")

#Feature engineering - create more meaningful features for fire prediction
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
   
   df['latitude_abs'] = np.abs(df['lat']) #Distance from equator
   return df

df = adjust_features(df)

#Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

#Encode categorical features
label_encoder = LabelEncoder()
df['weather_main_encoded'] = label_encoder.fit_transform(df['weather_main'].fillna('Unknown'))

#Select features for the model (expanded feature set)
features = [
    'temperature', 'humidity', 'wind_speed', 'pressure',
    'temp_min', 'temp_max', 'feels_like',
    'wind_gust_filled', 'clouds_pct', 'visibility_m',
    'rain_1h_mm', 'rain_3h_mm', 'snow_1h_mm', 'snow_3h_mm',
    
    'temp_range', 'is_hot', 'is_dry', 'humidity_temp_ratio',
    'is_windy', 'total_precip', 'has_recent_precip',
    'is_high_pressure', 'fire_danger_index', #Adjusted features
    
    'lat', 'lon', 'latitude_abs', #Geographic features
    
    'weather_main_encoded' #Encoded categorical features
]

#Features (X) vs Label (y)
#Features (independent variables): Inputs the model uses to predict.
#Label (dependent variable): The target we want to predict.
X = df[features]
y = df['historical_fire'].astype(int)

# Split to evaluate model performance on unseen data (avoid overfitting).
# X_train: Features for training (80% of data).
# X_test: Features for testing (20% of data).
# y_train: Labels for training.
# y_test: Labels for testing (ground truth for evaluation).
# random_state=42 ensures reproducibility (same split every time).
#stratify=y maintains class balance in split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42, stratify=y)

#Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#Tune hyperparameters for better performance
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

# Save model components
joblib.dump(model, "model_components/fire_risk_model.pkl")
joblib.dump(scaler, "model_components/fire_risk_scaler.pkl")
joblib.dump(label_encoder, "model_components/weather_encoder.pkl")
joblib.dump(features, "model_components/model_features.pkl")
