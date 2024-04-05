import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess the data
def load_preprocess_data():
    # Load the sequences and labels from the GitHub repo
    SEQUENCES_URL = 'https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/sequences.txt'
    LABELS_URL = 'https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/labels.txt'

    sequences = requests.get(SEQUENCES_URL).text.split('\n')
    labels = requests.get(LABELS_URL).text.split('\n')

    # remove empty sequences
    sequences = list(filter(None, sequences))
    labels = list(filter(None, labels)) 
    return sequences, labels

# One-hot encode the data: convert nucleotide sequence to a matrix of shape (num_sequences, seq_length, 4) consumable by a neural network
def one_hot_encode(sequences, labels):
    integer_encoder = LabelEncoder()  
    one_hot_encoder = OneHotEncoder(categories='auto')   
    input_features = []

    for sequence in sequences:
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        input_features.append(one_hot_encoded.toarray())

    np.set_printoptions(threshold=40)
    input_features = np.stack(input_features)
  
    labels = np.array(labels).reshape(-1, 1)
    input_labels = one_hot_encoder.fit_transform(labels).toarray()

    return input_features, input_labels

# Split the data into training and test sets
def split_data(input_features, input_labels):
    train_features, test_features, train_labels, test_labels = train_test_split(input_features, input_labels, test_size=0.25, random_state=42)
    return train_features, test_features, train_labels, test_labels