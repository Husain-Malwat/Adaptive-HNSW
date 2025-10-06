import numpy as np
import heapq
import random
import json
import os
import pickle

class HNSWScratch:
    """
    A from-scratch implementation of the HNSW algorithm in pure Python and NumPy.
    This implementation is for educational purposes to understand the algorithm's mechanics
    and is not optimized for performance like C++ libraries (e.g., hnswlib).
    """

    def __init__(self, space='l2', dim=128):
        """
        Initializes the HNSW index.
        Args:
            space (str): The metric space, one of 'l2', 'ip' (inner product), or 'cosine'.
            dim (int): The dimensionality of the vectors.
        """
        if space not in ['l2', 'ip', 'cosine']:
            raise ValueError("Space must be one of 'l2', 'ip', or 'cosine'")

        self.space = space
        self.dim = dim
        self._distance_func = self._get_distance_func(space)

        # Index data structures
        self.vectors = None
        self.graph = []  # List of dictionaries: {layer -> [neighbors]}
        self.node_layers = [] # Max layer for each node
        self.entry_point = None
        self.max_layer = -1
        self.element_count = 0
        
        # Index parameters
        self.M = 0
        self.ef_construction = 0
        self.mL = 0

    def _get_distance_func(self, space):
        """Returns the appropriate distance function for the space."""
        if space == 'l2':
            return lambda v1, v2: np.sum((v1 - v2)**2)
        if space == 'ip':
            # HNSW works with distances (lower is better), so we convert similarity to distance
            return lambda v1, v2: 1.0 - np.dot(v1, v2)
        if space == 'cosine':
            # Convert cosine similarity to cosine distance
            return lambda v1, v2: 1.0 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def init_index(self, max_elements, M=16, ef_construction=200):
        """
        Initializes the index with pre-allocated memory and parameters.
        Args:
            max_elements (int): The maximum number of elements the index will hold.
            M (int): The maximum number of connections per node per layer.
            ef_construction (int): The size of the dynamic candidate list during construction.
        """
        self.M = M
        self.ef_construction = ef_construction
        self.mL = 1 / np.log(self.M) # Normalization factor for random layer assignment
        
        # Pre-allocate memory
        self.vectors = np.zeros((max_elements, self.dim), dtype=np.float32)
        self.graph = [{} for _ in range(max_elements)]
        self.node_layers = [-1] * max_elements
        print(f"Index initialized for {max_elements} elements.")

    def _assign_layer(self):
        """Assigns a random layer to a new node based on an exponential distribution."""
        return int(-np.log(random.random()) * self.mL)

    def _search_layer_ef(self, query_vec, entry_points, ef, layer):
        """
        Searches a single layer to find the `ef` nearest neighbors to the query vector.
        This is the core search primitive used in both construction and querying.
        """
        candidates = set(entry_points)
        visited = set(entry_points)
        
        # Min-heap for candidates (stores (distance, node_id))
        candidate_heap = []
        for ep in entry_points:
            dist = self._distance_func(query_vec, self.vectors[ep])
            heapq.heappush(candidate_heap, (dist, ep))
            
        # Max-heap for results (stores (-distance, node_id) to simulate max-heap)
        result_heap = []
        for ep in entry_points:
            dist = self._distance_func(query_vec, self.vectors[ep])
            heapq.heappush(result_heap, (-dist, ep))

        while candidate_heap:
            dist, current_node_id = heapq.heappop(candidate_heap)

            # If current node is farther than the worst result, we can stop
            if result_heap and -result_heap[0][0] < dist:
                break
            
            # Explore neighbors of the current node
            neighbors = self.graph[current_node_id].get(layer, [])
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_dist = self._distance_func(query_vec, self.vectors[neighbor_id])
                    
                    if len(result_heap) < ef or -result_heap[0][0] > neighbor_dist:
                        heapq.heappush(candidate_heap, (neighbor_dist, neighbor_id))
                        heapq.heappush(result_heap, (-neighbor_dist, neighbor_id))
                        if len(result_heap) > ef:
                            heapq.heappop(result_heap)
        
        return [(node_id, -dist) for dist, node_id in sorted(result_heap, key=lambda x: -x[0])]

    def add_items(self, data, ids):
        """Adds a batch of items to the index."""
        if self.vectors is None:
            raise RuntimeError("Index not initialized. Call init_index first.")
            
        for i, node_id in enumerate(ids):
            vector = data[i]
            self.vectors[node_id] = vector
            
            # Assign a random layer for the new node
            new_node_layer = self._assign_layer()
            self.node_layers[node_id] = new_node_layer
            
            # Handle the very first element
            if self.entry_point is None:
                self.entry_point = node_id
                self.max_layer = new_node_layer
                self.element_count += 1
                continue

            # --- Stage 1: Top-down search for entry points ---
            current_node_id = self.entry_point
            # Search from the global max layer down to the layer above the new node's layer
            for l in range(self.max_layer, new_node_layer, -1):
                # Simple greedy search to find the best entry point for the next layer
                candidates = self._search_layer_ef(vector, [current_node_id], 1, l)
                current_node_id = candidates[0][0]

            # --- Stage 2: Connect neighbors from new node's layer down to 0 ---
            entry_points = [current_node_id]
            for l in range(min(new_node_layer, self.max_layer), -1, -1):
                # Find ef_construction nearest neighbors in the current layer
                neighbors = self._search_layer_ef(vector, entry_points, self.ef_construction, l)
                
                # Select M best neighbors
                selected_neighbors = [n[0] for n in neighbors[:self.M]]
                
                # Establish bidirectional connections
                self.graph[node_id][l] = selected_neighbors
                for neighbor_id in selected_neighbors:
                    neighbor_connections = self.graph[neighbor_id].get(l, [])
                    neighbor_connections.append(node_id)

                    # Pruning: if a neighbor has > M connections, keep the M best
                    if len(neighbor_connections) > self.M:
                        distances = [self._distance_func(self.vectors[neighbor_id], self.vectors[conn]) for conn in neighbor_connections]
                        sorted_conns = [x for _, x in sorted(zip(distances, neighbor_connections))]
                        self.graph[neighbor_id][l] = sorted_conns[:self.M]
                    else:
                        self.graph[neighbor_id][l] = neighbor_connections
                
                entry_points = [n[0] for n in neighbors]

            # Update the global entry point if necessary
            if new_node_layer > self.max_layer:
                self.max_layer = new_node_layer
                self.entry_point = node_id
                
            self.element_count += 1

    def knn_query(self, query_data, k, ef_search=None):
        """
        Performs KNN query for a batch of vectors.
        For simplicity, this implementation handles one query vector at a time.
        """
        if self.entry_point is None:
            return [], []

        # Assuming query_data is a single vector for this implementation
        query_vec = query_data[0] if query_data.ndim == 2 else query_data
        
        current_node_id = self.entry_point
        # Stage 1: Top-down search to find entry point for the bottom layer
        for l in range(self.max_layer, 0, -1):
            candidates = self._search_layer_ef(query_vec, [current_node_id], 1, l)
            current_node_id = candidates[0][0]
            
        # Stage 2: Perform detailed search on the bottom layer (layer 0)
        ef = ef_search if ef_search is not None else self.ef_construction
        neighbors = self._search_layer_ef(query_vec, [current_node_id], ef, 0)
        
        # Return top k results
        results = neighbors[:k]
        labels = np.array([[r[0] for r in results]])
        distances = np.array([[r[1] for r in results]])
        
        return labels, distances

    def save_index(self, path):
        """Saves the index to a specified path."""
        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        index_data = {
            'metadata': {
                'space': self.space,
                'dim': self.dim,
                'M': self.M,
                'ef_construction': self.ef_construction,
                'mL': self.mL,
                'element_count': self.element_count,
                'max_layer': self.max_layer,
                'entry_point': self.entry_point
            }
        }
        # Save metadata as JSON
        with open(path + '.json', 'w') as f:
            json.dump(index_data, f)
        
        # Save main data structures
        np.save(path + '.vectors.npy', self.vectors[:self.element_count])
        with open(path + '.graph.pkl', 'wb') as f:
            pickle.dump(self.graph[:self.element_count], f)
        with open(path + '.layers.pkl', 'wb') as f:
            pickle.dump(self.node_layers[:self.element_count], f)
        print(f"Index saved to {path}")

    def load_index(self, path):
        """Loads the index from a specified path."""
        # Load metadata
        with open(path + '.json', 'r') as f:
            index_data = json.load(f)
        
        meta = index_data['metadata']
        self.space = meta['space']
        self.dim = meta['dim']
        self.M = meta['M']
        self.ef_construction = meta['ef_construction']
        self.mL = meta['mL']
        self.element_count = meta['element_count']
        self.max_layer = meta['max_layer']
        self.entry_point = meta['entry_point']
        self._distance_func = self._get_distance_func(self.space)
        
        # Load main data structures
        loaded_vectors = np.load(path + '.vectors.npy')
        max_elements = len(loaded_vectors) # Or load from metadata if saved
        self.vectors = np.zeros((max_elements, self.dim), dtype=np.float32)
        self.vectors[:max_elements] = loaded_vectors
        
        with open(path + '.graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
        with open(path + '.layers.pkl', 'rb') as f:
            self.node_layers = pickle.load(f)
        print(f"Index loaded from {path}")