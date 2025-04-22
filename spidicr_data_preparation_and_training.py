
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# 1. Download Dataset (Manual Step Required)
print("Please download the CIC IoT-DIAD 2024 dataset from: https://www.unb.ca/cic/datasets/iot-diad.html")
print("After downloading, extract and place the CSV files into a folder named 'CIC_IOT_DIAD_2024'.")

data_dir = 'CIC_IOT_DIAD_2024'
if not os.path.exists(data_dir):
    raise FileNotFoundError("Data directory 'CIC_IOT_DIAD_2024' not found. Please follow the instructions above.")

# 2. Load Data
print("Loading dataset...")
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
df_list = [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

# 3. Basic Preprocessing
print("Preprocessing data...")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# 4. Encode labels (e.g., 'BENIGN', 'DoS', etc.)
if 'Label' not in df.columns:
    raise ValueError("Expected 'Label' column not found.")
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# 5. Feature Selection
X = df.drop(columns=['Label'])
y = df['Label']

# 6. Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Model Architecture (SpiderNet BiLSTM)
print("Training model...")
timesteps = 1
features = X_train.shape[1]
X_train = np.reshape(X_train, (X_train.shape[0], timesteps, features))
X_test = np.reshape(X_test, (X_test.shape[0], timesteps, features))

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(timesteps, features)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[early_stop])

# 9. Evaluation
print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
