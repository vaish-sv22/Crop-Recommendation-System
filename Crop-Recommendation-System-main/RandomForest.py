import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

# Sample dataset creation for the sake of illustration (replace with your actual dataset)
# Assuming the dataset has columns: 'pH', 'temperature', 'humidity', 'crop'
#data=pd.read_csv('Crop_recommendation')
csvFile = pd.read_csv(r'C:\Crop_recommendation.csv')



# Convert to DataFrame
df = pd.DataFrame(csvFile)

# Feature columns
X = df[['ph', 'temperature', 'humidity']]

# Target variable (Crop)
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')


# Example usage: Predict crop based on input values
input_data = np.array([[6.8, 24.3, 61.1]])  # Example: pH=6.8, Temp=28°C, Humidity=72%
crop_prediction = rf_model.predict(input_data)
print(f'The predicted crop is: {crop_prediction[0]}')
pickle.dump(rf_model,open('model.pkl','wb'))

