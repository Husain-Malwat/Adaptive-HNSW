import os
import numpy as np
from utils import fvecs_read, ivecs_read, generate_dummy_data

def load_data(config):
    """ Loads base, query, and ground truth vectors based on config """
    if config.get('generate_dummy_data', False):
        print("Using generated dummy data.")
        dummy_config = config['dummy_data_config']
        base_vectors, query_vectors, ground_truth = generate_dummy_data(
            dummy_config['num_base'],
            dummy_config['num_queries'],
            dummy_config['dim'],
            dummy_config['num_ground_truth']
        )
        dim = dummy_config['dim']
    else:
        print(f"Loading data from directory: {config['data_dir']}")
        base_path = os.path.join(config['data_dir'], config['base_file'])
        query_path = os.path.join(config['data_dir'], config['query_file'])
        gt_path = os.path.join(config['data_dir'], config['ground_truth_file'])

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base vector file not found: {base_path}")
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"Query vector file not found: {query_path}")

        base_vectors = fvecs_read(base_path)
        query_vectors = fvecs_read(query_path)
        dim = base_vectors.shape[1]

        if os.path.exists(gt_path):
            print(f"Loading ground truth from: {gt_path}")
            ground_truth = ivecs_read(gt_path)
            # Ensure ground truth K is not larger than what we have
            max_gt_k = ground_truth.shape[1]
            config['recall_k'] = [k for k in config['recall_k'] if k <= max_gt_k]
            print(f"Adjusted recall K values to be <= {max_gt_k}: {config['recall_k']}")
        else:
            print(f"Warning: Ground truth file not found: {gt_path}. Recall calculation will be skipped.")
            ground_truth = None

    print(f"Base vectors shape: {base_vectors.shape}")
    print(f"Query vectors shape: {query_vectors.shape}")
    if ground_truth is not None:
        print(f"Ground truth shape: {ground_truth.shape}")

    # Update config with detected dimension if not dummy data
    if not config.get('generate_dummy_data', False):
        config['hnsw_dim'] = dim

    return base_vectors, query_vectors, ground_truth