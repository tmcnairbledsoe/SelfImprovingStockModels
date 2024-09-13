import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

np.random.seed(42)
num_samples = 1000

original_inputs = np.random.rand(num_samples, 5)
predicted_prices = np.random.rand(num_samples, 1)

actions_taken = np.random.randint(0, 3, num_samples)

profit_loss = np.random.rand(num_samples, 1) * 200 - 100  

labels = (np.random.rand(num_samples) > 0.5).astype(int)

X = np.hstack((original_inputs, predicted_prices, actions_taken.reshape(-1, 1), profit_loss))

action_encoder = LabelEncoder()
X[:, -2] = action_encoder.fit_transform(X[:, -2])

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
