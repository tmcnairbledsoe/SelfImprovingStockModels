import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from datetime import timedelta

# Load your dataset (replace with actual data loading)
# For this example, we will generate dummy data
np.random.seed(42)
def generate_dummy_stock_data():
    start_time = pd.Timestamp('2024-01-01 09:30:00')
    time_index = pd.date_range(start=start_time, periods=78, freq='5T')
    data = pd.DataFrame({
        'timestamp': time_index,
        'open': np.random.rand(len(time_index)) * 100,
        'high': np.random.rand(len(time_index)) * 100,
        'low': np.random.rand(len(time_index)) * 100,
        'close': np.random.rand(len(time_index)) * 100,
        'volume': np.random.randint(1000, 10000, len(time_index))
    })
    return data

stock_data = generate_dummy_stock_data()

# Normalize the data
scaler = StandardScaler()
stock_data[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(stock_data[['open', 'high', 'low', 'close', 'volume']])

# Function to create a sequence of data for training
def create_sequences(data, start_time='09:30:00', end_time='13:00:00', target_end_time='16:00:00'):
    start_time = pd.Timestamp('2024-01-01 ' + start_time)
    end_time = pd.Timestamp('2024-01-01 ' + end_time)
    target_end_time = pd.Timestamp('2024-01-01 ' + target_end_time)
    
    start_idx = data.index[data['timestamp'] == start_time][0]
    end_idx = data.index[data['timestamp'] == end_time][0]
    target_end_idx = data.index[data['timestamp'] == target_end_time][0]

    # Randomly select a number of ticks between start_idx and end_idx
    random_idx = np.random.randint(start_idx, end_idx + 1)
    
    # Select input sequence and target sequence
    X = data.iloc[start_idx:random_idx].drop('timestamp', axis=1).values
    y = data.iloc[random_idx:target_end_idx + 1]['close'].values
    
    return X, y

# Generate the training and test data
X, y = create_sequences(stock_data)

# Reshape the data for the model (LSTM/GRU expect 3D input: samples, timesteps, features)
X = X.reshape((1, X.shape[0], X.shape[1]))  # single sample for demo purposes
y = y.reshape((1, y.shape[0]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create different models
def create_model(model_type='dense', input_shape=(None, 5), units=64, layers=2, dropout_rate=0.2):
    model = Sequential()
    
    if model_type == 'dense':
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(units, activation='relu'))
        for _ in range(layers - 1):
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'conv1d':
        model.add(Conv1D(filters=units, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(Flatten())
        for _ in range(layers - 1):
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'lstm':
        model.add(LSTM(units, return_sequences=(layers > 1), input_shape=input_shape))
        for _ in range(layers - 2):
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout_rate))
        model.add(LSTM(units))
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'gru':
        model.add(GRU(units, return_sequences=(layers > 1), input_shape=input_shape))
        for _ in range(layers - 2):
            model.add(GRU(units, return_sequences=True))
            model.add(Dropout(dropout_rate))
        model.add(GRU(units))
        model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    return model

# Wrap the model creation in a scikit-learn estimator
def build_model_fn(model_type='dense', units=64, layers=2, dropout_rate=0.2):
    input_shape = (None, X_train.shape[2])
    model = create_model(model_type=model_type, input_shape=input_shape, units=units, layers=layers, dropout_rate=dropout_rate)
    return model

# Pipeline for hyperparameter tuning and feature selection
model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model_fn, verbose=0)

param_dist = {
    'model_type': ['dense', 'conv1d', 'lstm', 'gru'],
    'units': [32, 64, 128],
    'layers': [1, 2, 3],
    'dropout_rate': [0.0, 0.2, 0.4],
    'epochs': [10, 50, 100],
    'batch_size': [16, 32, 64]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1)

# Use RFE for feature selection
rfe = RFE(estimator=random_search, n_features_to_select=3)

# Create a pipeline
pipeline = Pipeline([
    ('rfe', rfe)
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate on the test set
best_model = pipeline.named_steps['rfe'].estimator_.best_estimator_
test_loss, test_mae = best_model.model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {test_mae:.4f}")

# Summary of the best model
best_model.model.summary()
