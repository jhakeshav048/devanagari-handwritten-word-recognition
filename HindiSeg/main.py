import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle

# Load the 'data.txt' file with image paths and corresponding Hindi words
data_txt_path = 'HindiSeg/data.txt'
data = pd.read_csv(data_txt_path, sep=" ", header=None, names=["Image_Path", "Word"])

# Image Preprocessing Function
def preprocess_image(image_path, target_size=(64, 32)):
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(target_size)  # Resize to (64, 32)
        img_array = np.array(img) / 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Apply image preprocessing to each image path
data['Image_Array'] = data['Image_Path'].apply(lambda x: preprocess_image(os.path.join('HindiSeg', x)))
data = data.dropna(subset=['Image_Array'])

# Custom Tokenizer for Hindi characters
unique_chars = sorted(set(''.join(data['Word'])))
char_to_index = {char: idx + 1 for idx, char in enumerate(unique_chars)}

# Encode each Hindi word
def encode_word(word):
    return [char_to_index[char] for char in word]

data['Encoded_Word'] = data['Word'].apply(encode_word)

# Save character to index mapping
with open('char_to_index.pkl', 'wb') as f:
    pickle.dump(char_to_index, f)

# Pad Sequences
max_length = max(data['Encoded_Word'].apply(len))
data['Padded_Word'] = data['Encoded_Word'].apply(lambda x: x + [0] * (max_length - len(x)))

# Prepare data for model input
X = np.array(data['Image_Array'].tolist()).reshape(-1, 32, 64, 1)  # Reshape to (32, 64, 1)
y = np.array(data['Padded_Word'].tolist())

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
print("Preprocessing complete.")
