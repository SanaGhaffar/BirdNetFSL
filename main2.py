import math
from keras import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.random_projection import SparseRandomProjection
import warnings
import platform
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import os
import audiomentations as aa
from tqdm import tqdm
import librosa

# Suppress warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn  # Override warnings.warn with an empty function to suppress warnings

# Set random seed for reproducibility
random.seed(42)  # Set the random seed for random number generation

# Define constants
DATASET_DIRECTORY = "archive/songs"  # Directory containing the dataset
METADATA_FILE = 'archive/birdsong_metadata.csv'  # File path for metadata
MODEL_PATH = '.'  # Path for the model


# Function to load metadata from a file
def load_metadata(file_path):
    return pd.read_csv(file_path)  # Read metadata from a CSV file and return as a DataFrame


# Function to create a mapping between file IDs and labels
def create_mapping(metadata, birdnet_labels):
    mapping = {}  # Initialize an empty dictionary for mapping
    for index, row in metadata.iterrows():  # Iterate through each row in the metadata DataFrame
        file_id = row['file_id']  # Get the file ID
        label = f"{row['genus']} {row['species']}_{row['english_cname']}"  # Create a label
        if label not in birdnet_labels.values():  # Check if label is not present in birdnet_labels
            mapping[file_id] = label  # Add file ID and label to the mapping dictionary
    return mapping  # Return the mapping dictionary


# Function to load audio data based on directory, mapping, and batch size
def load_data(directory, mapping, batch_size):
    audios = list()  # Initialize an empty list to store audio data
    labels = list()  # Initialize an empty list to store labels
    augmenter = aa.Compose([  # Define a series of audio augmentations
        aa.AddGaussianNoise(p=0.2),
        aa.TimeStretch(p=0.2),
        aa.PitchShift(p=0.2),
    ])
    file_paths = list()  # Initialize an empty list to store file paths

    # Traverse the directory to find audio files
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)  # Construct file path
            if platform.system() == 'Windows':
                file_id = file.split('\\')[-1].split('.')[0]  # Extract file ID for Windows
            else:
                file_id = file.split('/')[-1].split('.')[0]  # Extract file ID for other platforms
            file_id = int(file_id[2:])  # Convert file ID to integer
            if file_id in mapping:  # Check if file ID is in the mapping
                file_paths.append(file_path)  # Add file path to the list

    num_batches = math.ceil(len(file_paths) / batch_size)  # Calculate the number of batches
    random_projection = SparseRandomProjection(n_components=144000)  # Initialize SparseRandomProjection

    # Load audio data in batches
    for batch_num in tqdm(range(num_batches), desc='LOADING DATA...'):  # Iterate through each batch
        batch_audios = []  # Initialize a list to store batched audio data
        batch_labels = []  # Initialize a list to store batched labels
        offset = batch_num * batch_size  # Calculate the starting index of the batch
        batch_files = file_paths[offset:offset + batch_size]  # Get files for the current batch

        # Process each file in the batch
        for file_path in batch_files:
            audio, sample_rate = librosa.load(file_path, sr=None, mono=False)  # Load audio file
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)  # Convert stereo audio to mono if applicable
            if platform.system() == 'Windows':
                file_id = file_path.split('\\')[-1].split('.')[0]  # Extract file ID for Windows
            else:
                file_id = file_path.split('/')[-1].split('.')[0]  # Extract file ID for other platforms
            file_id = int(file_id[2:])  # Convert file ID to integer
            label = mapping[file_id]  # Get label from mapping
            audio = audio.astype(np.float16)  # Convert audio data to float16
            batch_audios.append(audio)  # Add audio data to the batch
            batch_labels.append(label)  # Add label to the batch

        # Pad audio data to have equal lengths within a batch
        max_length = max(len(arr) for arr in batch_audios)  # Find the maximum length in the batch
        for i in range(len(batch_audios)):
            arr = batch_audios[i]  # Get individual audio array
            padded_audio = np.pad(arr, (0, max_length - len(arr)))  # Pad audio to match max_length
            batch_audios[i] = padded_audio  # Update batch audio with padded audio

        # Transform and convert batch audios using random projection
        batch_audios = np.array(batch_audios)  # Convert batch audios to numpy array
        batch_audios = random_projection.fit_transform(batch_audios).tolist()  # Perform random projection
        audios.extend(batch_audios)  # Add transformed audios to the audios list
        labels.extend(batch_labels)  # Add labels to the labels list

    return audios, labels  # Return the audios and labels


# Function to perform zero-shot classification
def do_zero_shot(birdnet_model, audios, labels, birdnet_labels):
    # Obtain embeddings for seen classes from the BirdNET model
    seen_class_embeddings = birdnet_model.layers[-1].get_weights()[0].T  # Get weights for seen classes
    seen_class_semantic_space = {}  # Initialize dictionary for seen class embeddings

    # Create a dictionary for seen class embeddings normalized to unit vectors
    for i, embedding in enumerate(seen_class_embeddings):
        label = birdnet_labels[i]  # Get label from birdnet_labels
        seen_class_semantic_space[label] = embedding / np.linalg.norm(embedding)  # Normalize and store embedding

    # Create a model with the last hidden layer of the BirdNET model
    last_hidden_layer_model = Model(inputs=birdnet_model.input, outputs=birdnet_model.layers[-2].output)
    predicted_labels = []  # Initialize a list to store predicted labels

    # Predict labels for input audios
    for audio, target_label in zip(audios, labels):
        hidden_activations = last_hidden_layer_model.predict(np.expand_dims(audio, axis=0),
                                                             verbose=0)  # Get hidden activations

        # Find the most similar seen class embedding for each audio
        max_similarity = -math.inf  # Initialize maximum similarity
        for label, vector in seen_class_semantic_space.items():
            vector = vector.reshape(1, -1)  # Reshape vector for cosine similarity calculation
            similarity = cosine_similarity(hidden_activations, vector)[0][0]  # Calculate cosine similarity
            if similarity > max_similarity:
                max_similarity = similarity  # Update max similarity
                predicted_label = label  # Update predicted label

        predicted_labels.append(predicted_label)  # Append predicted label

    # Count occurrences of predicted labels
    predicted_labels = {element: predicted_labels.count(element) for element in predicted_labels}
    print(f"Predicted Labels: {predicted_labels}")  # Print predicted labels and their counts


# Main function
def main():
    # Load metadata and create mapping
    metadata = load_metadata(METADATA_FILE)  # Load metadata from file
    birdnet_labels = {i: line.strip() for i, line in
                      enumerate(open('BirdNET_GLOBAL_6K_V2.4_Labels_af.txt', 'r'))}  # Create dictionary from file
    mapping = create_mapping(metadata, birdnet_labels)  # Create mapping between file IDs and labels
    batch_size = 10  # Set batch size
    audios, labels = load_data(DATASET_DIRECTORY, mapping, batch_size)  # Load audio data

    # Load the BirdNET model and perform zero-shot classification
    birdnet_model = tf.keras.models.load_model(MODEL_PATH).model  # Load BirdNET model
    print('**************************************************')
    print('Zero Shot')
    print('**************************************************')
    do_zero_shot(birdnet_model, audios, labels, birdnet_labels)  # Perform zero-shot classification


if __name__ == "__main__":
    main()  # Call the main function if the script is executed directly

