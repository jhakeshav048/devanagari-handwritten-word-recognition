import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics import confusion_matrix

# Load the model and data
model = load_model('final_model.keras')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# Load the character-to-index mapping
with open('char_to_index.pkl', 'rb') as f:
    char_to_index = pickle.load(f)
index_to_char = {v: k for k, v in char_to_index.items()}

# Decode sequences
def decode_sequence(sequence):
    return ''.join([index_to_char.get(int(idx), '') for idx in sequence if idx != 0])

# Get model predictions
predictions = model.predict(X_val)
predicted_labels = np.argmax(predictions, axis=-1)

# 1. Sample Predictions Visualization
def plot_sample_predictions(num_samples=5):
    plt.figure(figsize=(15, num_samples * 3))
    sample_indices = np.random.choice(len(X_val), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        img = X_val[idx].reshape(32, 64)  # Adjust if image shape differs
        true_text = decode_sequence(y_val[idx])
        pred_text = decode_sequence(predicted_labels[idx])
        
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_text} | Pred: {pred_text}", color='green' if true_text == pred_text else 'red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 2. Confusion Matrix (Character-Level)
def plot_character_confusion_matrix():
    true_chars, pred_chars = [], []
    
    for i in range(len(y_val)):
        true_sequence = y_val[i]
        pred_sequence = predicted_labels[i]
        
        for true_char, pred_char in zip(true_sequence, pred_sequence):
            if true_char != 0:  # Ignore padding
                true_chars.append(true_char)
                pred_chars.append(pred_char)
    
    cm = confusion_matrix(true_chars, pred_chars)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=[index_to_char[i] for i in range(1, len(index_to_char) + 1)],
                yticklabels=[index_to_char[i] for i in range(1, len(index_to_char) + 1)])
    plt.xlabel('Predicted Characters')
    plt.ylabel('True Characters')
    plt.title('Character-Level Confusion Matrix')
    plt.show()

# 3. Edit Distance Histogram
def plot_edit_distance_histogram():
    edit_distances = []
    
    for i in range(len(y_val)):
        true_text = decode_sequence(y_val[i])
        pred_text = decode_sequence(predicted_labels[i])
        edit_distances.append(levenshtein_distance(true_text, pred_text))
    
    plt.figure(figsize=(10, 6))
    plt.hist(edit_distances, bins=20, color='purple', edgecolor='black')
    plt.xlabel('Edit Distance')
    plt.ylabel('Frequency')
    plt.title('Edit Distance Histogram')
    plt.show()

# Run the visualizations
print("Sample Predictions:")
plot_sample_predictions(num_samples=5)

print("Character-Level Confusion Matrix:")
plot_character_confusion_matrix()

print("Edit Distance Histogram:")
plot_edit_distance_histogram()
