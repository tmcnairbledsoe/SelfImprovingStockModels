import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

# Load or create your dataset (replace with actual data loading)
# For this example, we generate a dummy dataset
np.random.seed(42)
num_samples = 1000

# Assume original inputs are 5 features
original_inputs = np.random.rand(num_samples, 5)
predicted_prices = np.random.rand(num_samples, 1)

# Actions: 0 = hold, 1 = buy, 2 = sell
actions_taken = np.random.randint(0, 3, num_samples)

# Cumulative profit/loss
profit_loss = np.random.rand(num_samples, 1) * 200 - 100  # Ranges between -100 and +100

# Label: 0 = bad decision, 1 = good decision
# For simplicity, let's assume if profit_loss increases after action, it was a good decision
labels = (np.random.rand(num_samples) > 0.5).astype(int)

# Combine all features
X = np.hstack((original_inputs, predicted_prices, actions_taken.reshape(-1, 1), profit_loss))

# Encode actions as categorical variables
action_encoder = LabelEncoder()
X[:, -2] = action_encoder.fit_transform(X[:, -2])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model creation
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Analysis and improvement suggestions
# This can include comparing predicted vs actual labels, analyzing false positives/negatives, etc.
