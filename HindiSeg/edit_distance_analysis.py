import numpy as np
import pickle
from tensorflow.keras.models import load_model
from Levenshtein import distance as levenshtein_distance  # Install with `pip install python-Levenshtein`

# Load the saved model and preprocessed data
model = load_model('final_model.keras')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# Load the character-to-index mapping
with open('char_to_index.pkl', 'rb') as f:
    char_to_index = pickle.load(f)
index_to_char = {v: k for k, v in char_to_index.items()}

# Function to decode predictions
def decode_sequence(predicted_sequence):
    return ''.join([index_to_char.get(int(idx), '') for idx in predicted_sequence if idx != 0])

# Get model predictions
predictions = model.predict(X_val)
predicted_labels = np.argmax(predictions, axis=-1)  # Get the most likely class for each timestep

# Calculate edit distance for each sequence
total_distance = 0
num_samples = len(y_val)

for i in range(num_samples):
    true_label = decode_sequence(y_val[i])
    pred_label = decode_sequence(predicted_labels[i])
    total_distance += levenshtein_distance(true_label, pred_label)

# Average edit distance
average_edit_distance = total_distance / num_samples
print(f"Average Levenshtein Distance: {average_edit_distance:.2f}")
