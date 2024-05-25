import numpy as np
import pandas as pd
import librosa
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

# Function to extract Mel-spectrograms from WAV files with optimizations
def extract_features(file_path, sr=22050, n_mels=32, duration=2):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)  # Load audio with the desired sampling rate
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Match files by ID
def find_file_by_id(data_dir, file_id):
    possible_filenames = [f for f in os.listdir(data_dir) if f.startswith(str(file_id))]
    if possible_filenames:
        return os.path.join(data_dir, possible_filenames[0])
    return None

# Define the CNN model creation function with regularization
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adjusted learning rate
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to classify a single track
def classify_track(model, file_path, max_len, genres):
    features = extract_features(file_path)
    if features is not None:
        features_padded = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')  # Pad to the same length as training data
        features_padded = features_padded[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        prediction = model.predict(features_padded)
        predicted_genre = genres[np.argmax(prediction)]
        return predicted_genre
    else:
        return "Unknown"

# Load data from CSV
csv_file = '/Users/cj/Downloads/csvfile.csv'
data_dir = '/Users/cj/Downloads/output_folder1'
df = pd.read_csv(csv_file)

# Prepare training data
X = []
y = []

# Assuming the columns are 'id', 'title', 'artists', 'mix', and 'genres'
id_col = 'id'
genre_col = 'genres'

# Extract genres from CSV
genres = df[genre_col].unique().tolist()

# Determine the maximum length for padding
max_len = 0
features_list = []

for index, row in df.iterrows():
    file_id = row[id_col]
    file_path = find_file_by_id(data_dir, file_id)
    
    if file_path and os.path.isfile(file_path):
        features = extract_features(file_path)
        if features is not None:
            max_len = max(max_len, features.shape[1])
            features_list.append(features)
            y.append(genres.index(row[genre_col]))
    else:
        print(f"File not found for ID: {file_id}")

# Pad sequences to the same length in batches
X_padded = np.zeros((len(features_list), features_list[0].shape[0], max_len), dtype='float32')

for i, features in enumerate(features_list):
    X_padded[i, :, :features.shape[1]] = features

# Normalize the data
X_padded = (X_padded - np.mean(X_padded)) / np.std(X_padded)

# Check if X and y are populated
if len(X_padded) == 0 or len(y) == 0:
    print("No valid samples found. Please check the file paths and data.")
else:
    X_padded = np.array(X_padded)
    y = np.array(y)

    # Reshape for CNN input
    X_padded = X_padded[..., np.newaxis]

    # Convert labels to categorical
    y = to_categorical(y, num_classes=len(genres))

    # Implement stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cvscores = []

    for train, test in skf.split(X_padded, np.argmax(y, axis=1)):
        # Create model
        model = create_cnn_model((X_padded.shape[1], X_padded.shape[2], 1), len(genres))
        
        # Early stopping and learning rate reduction callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

        # Fit the model
        model.fit(X_padded[train], y[train], epochs=30, batch_size=64, verbose=1,  # Adjusted epochs
                  validation_data=(X_padded[test], y[test]), callbacks=[early_stopping, lr_reduction])
        
        # Evaluate the model
        scores = model.evaluate(X_padded[test], y[test], verbose=0)
        print(f'Fold accuracy: {scores[1] * 100:.2f}%')
        cvscores.append(scores[1] * 100)

    print(f'Mean accuracy: {np.mean(cvscores):.2f}%')
    print(f'Standard deviation: {np.std(cvscores):.2f}%')

    # Load the three tracks and classify
    track1 = '/Users/cj/Documents/model_training/melody.wav'
    track2 = '/Users/cj/Documents/model_training/velocity.wav'
    track3 = '/Users/cj/Documents/model_training/retro.wav'

    # Assuming the last trained model is used for classification
    genre1 = classify_track(model, track1, max_len, genres)
    genre2 = classify_track(model, track2, max_len, genres)
    genre3 = classify_track(model, track3, max_len, genres)

    print(f'Track 1: {genre1}')
    print(f'Track 2: {genre2}')
    print(f'Track 3: {genre3}')
