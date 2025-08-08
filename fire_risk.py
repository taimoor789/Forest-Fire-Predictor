import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score  #Evaluation metrics
import joblib  #To save/load trained models


#Load the labeled data
df = pd.read_csv("labeled_weather_data.csv")

#Features (X) vs Label (y)
#Features (independent variables): Inputs the model uses to predict.
#Label (dependent variable): The target we want to predict.
features = ["temperature", "humidity", "wind_speed"]
X = df[features]
y = df["historical_fire"].astype(float)

# Split to evaluate model performance on unseen data (avoid overfitting).
# X_train: Features for training (80% of data).
# X_test: Features for testing (20% of data).
# y_train: Labels for training.
# y_test: Labels for testing (ground truth for evaluation).
# random_state=42 ensures reproducibility (same split every time).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#RandomForestRegressor: Ensemble model for regression (predicts continuous fire risk probability).
#n_estimators=100: Number of decision trees in the forest.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Train on training data (features + labels)

#Predict fire risk probabilities for test set features (X_test).
y_pred = model.predict(X_test)

#Mean Squared Error (MSE): Average squared difference between predicted/actual values (lower = better).
# R² Score: Proportion of variance explained by the model (1.0 = perfect fit).
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

#joblib.dump(): Saves the trained model to a file for later use (e.g., deployment).
joblib.dump(model, "fire_risk_model.pkl")
print("\nModel saved as fire_risk_model.pkl")