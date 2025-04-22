
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Function to plot reconstruction error histogram
def plot_reconstruction_error(errors, threshold):
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title("FrogTrigger Reconstruction Error")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reconstruction_error_plot.png")
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
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

# Assume 'recon_errors', 'threshold', 'y_test', 'predicted_labels', and 'predictions' already exist from main pipeline

# Visualize
plot_reconstruction_error(recon_errors, threshold)
plot_confusion_matrix(y_test, predicted_labels)
plot_roc_curve(y_test, predictions)
