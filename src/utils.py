import numpy as np
import struct
import os
import json
from tqdm import tqdm

def ivecs_read(fname):
    """ Reads .ivecs file format """
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    """ Reads .fvecs file format """
    return ivecs_read(fname).view('float32')

def ensure_dir(directory):
    """ Ensures a directory exists, creating it if necessary """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_json(data, filepath):
    """ Saves data to a JSON file """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {filepath}")

def load_json(filepath):
    """ Loads data from a JSON file """
    if not os.path.exists(filepath):
        print(f"Warning: JSON file not found at {filepath}")
        return None
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filepath}")
            return None

def append_jsonl(data, filepath):
    """ Appends a dictionary (as a JSON line) to a .jsonl file """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'a') as f:
        json.dump(data, f)
        f.write('\n')

def generate_dummy_data(num_base, num_queries, dim, num_ground_truth_k):
    """Generates random data for testing"""
    print(f"Generating dummy data: {num_base} base, {num_queries} queries, dim={dim}")
    base_vectors = np.float32(np.random.rand(num_base, dim))
    query_vectors = np.float32(np.random.rand(num_queries, dim))

    # Generate dummy ground truth (expensive for large data, simplified here)
    print(f"Generating dummy ground truth (finding actual neighbors)...")
    # Using a simple approach; for large data, ANN would be needed here too!
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=num_ground_truth_k, algorithm='brute', metric='euclidean')
    nn.fit(base_vectors)
    _, ground_truth_indices = nn.kneighbors(query_vectors)
    print("Dummy ground truth generation complete.")

    return base_vectors, query_vectors, ground_truth_indices