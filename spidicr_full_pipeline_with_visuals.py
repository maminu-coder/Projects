
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# === Load Placeholder Dataset (can replace with CIC IoT-DIAD) ===
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

env_features = X.iloc[:, :15]
behav_features = X.iloc[:, 15:]

env_scaler = StandardScaler()
env_scaled = env_scaler.fit_transform(env_features)

behav_scaler = StandardScaler()
behav_scaled = behav_scaler.fit_transform(behav_features)

X_env_train, X_env_test, X_behav_train, X_behav_test, y_train, y_test = train_test_split(
    env_scaled, behav_scaled, y, test_size=0.2, random_state=42
)

# === FROGTRIGGER (Autoencoder) ===
input_dim = X_env_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_env_train, X_env_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

recon_errors = np.mean(np.square(X_env_test - autoencoder.predict(X_env_test)), axis=1)
threshold = np.percentile(recon_errors, 95)
suspicious = recon_errors > threshold

# === SPIDERNET (BiLSTM) ===
X_behav_train_seq = np.reshape(X_behav_train, (X_behav_train.shape[0], 1, X_behav_train.shape[1]))
X_behav_test_seq = np.reshape(X_behav_test, (X_behav_test.shape[0], 1, X_behav_test.shape[1]))

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, X_behav_train.shape[1])))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_behav_train_seq, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# === HUNT PHASE ===
predictions = model.predict(X_behav_test_seq)
predicted_labels = (predictions > 0.5).astype(int)

print("SPIDICR Final Evaluation Report:")
print(classification_report(y_test, predicted_labels))

# === VISUALIZATION ===

# Reconstruction Error Plot
plt.figure(figsize=(8, 5))
plt.hist(recon_errors, bins=50, alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("FrogTrigger Reconstruction Error")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("reconstruction_error_plot.png")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
