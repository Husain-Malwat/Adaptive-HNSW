import hnswlib
import os
import time
import numpy as np
from tqdm import tqdm
from utils import ensure_dir

class HNSWWrapper:
    def __init__(self, config, logger=None):
        self.config = config
        self.space = config['hnsw_space']
        self.dim = config['hnsw_dim']
        self.index_path = config['hnsw_index_path']
        self.index = None
        self.logger = logger
        self.is_initialized = False
        print(f"HNSW Wrapper configured: space={self.space}, dim={self.dim}")

    def build_index(self, data, force_rebuild=False):
        """ Builds or loads the HNSW index """
        ensure_dir(os.path.dirname(self.index_path))

        if not force_rebuild and os.path.exists(self.index_path):
            print(f"Loading existing index from {self.index_path}")
            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            self.index.load_index(self.index_path)
            self.is_initialized = True
            print("Index loaded successfully.")
            # Optional: Check if data dimensions match index dimensions
            # print(f"Index max elements: {self.index.get_max_elements()}")
            # print(f"Index current count: {self.index.get_current_count()}")
            return

        print("Building new HNSW index...")
        num_elements = data.shape[0]
        ids = np.arange(num_elements)

        # Declaring index
        self.index = hnswlib.Index(space=self.space, dim=self.dim)

        # Initializing index - the maximum number of elements should be known beforehand
        self.index.init_index(max_elements=num_elements,
                              ef_construction=self.config['hnsw_ef_construction'],
                              M=self.config['hnsw_m'])

        # Add data points with progress bar
        print("Adding data points to the index...")
        batch_size = 10000 # Process in batches for potentially large datasets
        for i in tqdm(range(0, num_elements, batch_size)):
            end_idx = min(i + batch_size, num_elements)
            self.index.add_items(data[i:end_idx], ids[i:end_idx])

        print(f"Index built with {self.index.get_current_count()} elements.")

        # Save index
        print(f"Saving index to {self.index_path}")
        self.index.save_index(self.index_path)
        self.is_initialized = True
        print("Index build and save complete.")

    def search(self, query_vector, k, ef_search):
        """ Performs KNN search and logs the query if logger is attached """
        if not self.is_initialized or self.index is None:
            raise RuntimeError("Index is not built or loaded.")

        start_time = time.time()
        # Ensure query is a numpy array
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
             query_vector = np.expand_dims(query_vector, axis=0) # Reshape single vector

        # Set efSearch parameter dynamically
        self.index.set_ef(ef_search)

        # Perform the search
        labels, distances = self.index.knn_query(query_vector, k=k)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Log the query details if a logger is provided
        if self.logger:
            # Assuming single query for logging simplicity here
            self.logger.log(query_vector[0], labels[0], distances[0], latency_ms, ef_search)

        return labels[0], distances[0], latency_ms # Return results for the first query
    
    
    
    
    
    
    
    
    
    
    
    
#     #----------------------------------------------------
# # File: adaptive-hnsw/src/hnsw_wrapper.py
# #----------------------------------------------------
# # import hnswlib # We no longer need this library
# from hnsw_from_scratch import HNSWScratch # Import our new class
# import os
# import time
# import numpy as np
# from tqdm import tqdm
# from utils import ensure_dir

# class HNSWWrapper:
#     def __init__(self, config, logger=None):
#         self.config = config
#         self.space = config['hnsw_space']
#         self.dim = config['hnsw_dim']
#         self.index_path = config['hnsw_index_path']
#         self.logger = logger
#         self.is_initialized = False

#         # Use our from-scratch implementation
#         self.index = HNSWScratch(space=self.space, dim=self.dim)
#         print(f"HNSW Wrapper (from-scratch) configured: space={self.space}, dim={self.dim}")

#     def build_index(self, data, force_rebuild=False):
#         """ Builds or loads the HNSW index using the from-scratch implementation """
#         ensure_dir(os.path.dirname(self.index_path))

#         # Check for saved index files (note the new extensions)
#         if not force_rebuild and os.path.exists(self.index_path + '.json'):
#             print(f"Loading existing index from {self.index_path}")
#             self.index.load_index(self.index_path)
#             self.is_initialized = True
#             print("Index loaded successfully.")
#             print(f"Index current count: {self.index.element_count}")
#             return

#         print("Building new HNSW index (from-scratch)...")
#         num_elements = data.shape[0]
#         ids = np.arange(num_elements)

#         # Initializing index
#         self.index.init_index(max_elements=num_elements,
#                               ef_construction=self.config['hnsw_ef_construction'],
#                               M=self.config['hnsw_m'])

#         # Add data points with progress bar
#         print("Adding data points to the index...")
#         # Note: The from-scratch version is slow, so batching here is for progress display,
#         # not performance. The internal loop is one-by-one.
#         for i in tqdm(range(num_elements)):
#             # Our add_items takes a batch, but we'll call it for each item
#             # to show progress that matches the slow insertion speed.
#             self.index.add_items(np.expand_dims(data[i], axis=0), [ids[i]])

#         print(f"Index built with {self.index.element_count} elements.")

#         # Save index
#         print(f"Saving index to {self.index_path}")
#         self.index.save_index(self.index_path)
#         self.is_initialized = True
#         print("Index build and save complete.")

#     def search(self, query_vector, k, ef_search):
#         """ Performs KNN search and logs the query if logger is attached """
#         if not self.is_initialized or self.index is None:
#             raise RuntimeError("Index is not built or loaded.")

#         start_time = time.time()
        
#         if not isinstance(query_vector, np.ndarray):
#             query_vector = np.array(query_vector, dtype=np.float32)
#         if query_vector.ndim == 1:
#             query_vector = np.expand_dims(query_vector, axis=0)

#         # In our implementation, ef_search is a direct argument to knn_query
#         labels, distances = self.index.knn_query(query_vector, k=k, ef_search=ef_search)
#         end_time = time.time()
#         latency_ms = (end_time - start_time) * 1000

#         if self.logger:
#             self.logger.log(query_vector[0], labels[0], distances[0], latency_ms, ef_search)

#         return labels[0], distances[0], latency_ms