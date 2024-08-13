import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv('dataset.csv')

# Preprocessing
label_encoders = {}
for column in ['track', 'weather', 'car_setup']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data[['car_no', 'track', 'track_length', 'no_of_laps', 'weather', 'temperature', 'start_pos', 'car_setup']]
y = data['SUGGESTED_STRATEGY']

# Check for NaN in X and y
print("NaN in each column of X before handling:")
print(X.isnull().sum())
print("\nNaN in y before handling:")
print(y.isnull().sum())

# Handle NaN in X
imputer = SimpleImputer(strategy='mean')  # Change strategy as needed
X_imputed = imputer.fit_transform(X)

# Handle NaN in y
if y.isnull().sum() > 0:
    y = y.fillna(y.mode()[0])  # Fill NaN in y

# Ensure data consistency
if X_imputed.shape[0] != y.shape[0]:
    raise ValueError("Mismatch between X and y dimensions after handling NaNs.")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check for NaN in the training data
print("NaN in each column of X_train after handling:")
print(pd.DataFrame(X_train).isnull().sum())
print("\nNaN in y_train after handling:")
print(pd.Series(y_train).isnull().sum())

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Making a Prediction
def suggest_strategy(car_no, track, weather, temperature, start_pos, car_setup):
    track = label_encoders['track'].transform([track])[0]
    weather = label_encoders['weather'].transform([weather])[0]
    car_setup = label_encoders['car_setup'].transform([car_setup])[0]
    
    input_data = pd.DataFrame([[car_no, track, None, None, weather, temperature, start_pos, car_setup]], 
                              columns=['car_no', 'track', 'track_length', 'no_of_laps', 'weather', 'temperature', 'start_pos', 'car_setup'])
    
    # Impute missing values in input data
    input_data_imputed = imputer.transform(input_data)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data_imputed)
    
    return model.predict(input_data_scaled)

# Example Usage
strategy = suggest_strategy(1, 'Circuit de Monaco', 'Sunny', 30, 1, 'High downforce')
print("Suggested Strategy:", strategy)





import numpy as np
import matplotlib.pyplot as plt

def parse_strategy(strategy):
    # Split the strategy into individual stints
    stints = strategy.split(', ')
    return [(s.split('-')[0], int(s.split('-')[1])) for s in stints]

def plot_strategy(strategy_array):
    # Ensure strategy is a string
    if isinstance(strategy_array, np.ndarray):
        strategy = strategy_array[0]  # Extract the first element if it's an array
    else:
        strategy = strategy_array

    # Map each tire type to a color
    tire_colors = {
        'Soft': 'red',
        'Medium': 'yellow',
        'Hard': 'white',
        'Intermediate': 'green',
        'Wet': 'blue'
    }
    
    # Parse the strategy
    parsed_stints = parse_strategy(strategy)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 2))
    current_lap = 0

    for tire, laps in parsed_stints:
        ax.barh(0, laps, left=current_lap, color=tire_colors[tire], edgecolor='black')
        current_lap += laps

    # Add labels and title
    ax.set_xlim(0, sum(laps for _, laps in parsed_stints))
    ax.set_yticks([])
    ax.set_xlabel('Laps')
    ax.set_title('Race Strategy Visualization')

    # Show plot
    plt.show()



plot_strategy(strategy)
