import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the saved model and preprocessed data
model = load_model('final_model.keras')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# Load the character-to-index mapping
with open('char_to_index.pkl', 'rb') as f:
    char_to_index = pickle.load(f)
index_to_char = {v: k for k, v in char_to_index.items()}

# Get model predictions
predictions = model.predict(X_val)
predicted_labels = np.argmax(predictions, axis=-1)  # Get the most likely class for each timestep

# Calculate character-level accuracy
correct_characters = 0
total_characters = 0

for i in range(len(y_val)):
    true_sequence = y_val[i]
    pred_sequence = predicted_labels[i]
    
    # Count correct characters, ignoring padding (0)
    for true_char, pred_char in zip(true_sequence, pred_sequence):
        if true_char != 0:  # Ignore padding
            total_characters += 1
            if true_char == pred_char:
                correct_characters += 1

character_accuracy = correct_characters / total_characters * 100
print(f"Character-Level Accuracy: {character_accuracy:.2f}%")
