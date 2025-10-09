# Adaptive HNSW: Intelligent Vector Search with Query-Driven Optimization

> **A research-oriented implementation of Hierarchical Navigable Small World (HNSW) graphs with adaptive optimization strategies for approximate nearest neighbor search.**

This repository provides both a **custom implementation** and an **adaptive optimization framework** for HNSW, demonstrating how query patterns can be leveraged to improve search performance while maintaining accuracy.

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Theoretical Background](#-theoretical-background)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Benchmarking](#-benchmarking)
- [Adaptive Strategies](#-adaptive-strategies)
- [Performance Results](#-performance-results)
- [Datasets](#-datasets)
- [Configuration](#-configuration)
- [Architecture](#-architecture)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## Overview

**Adaptive HNSW** is a toolkit for approximate nearest neighbor (ANN) search that combines:

1. **Custom Implementation**: A pure Python/NumPy implementation of HNSW from scratch to understand the algorithm's internals
2. **Production Wrapper**: Integration with the high-performance [hnswlib](https://github.com/nmslib/hnswlib) C++ library
3. **Adaptive Optimization**: Query pattern analysis and index adaptation for improved performance
4. **Comprehensive Benchmarking**: Tools to compare implementations and measure performance across multiple metrics

### What Makes This Project Unique?

- 🎓 **Learning-Focused**: Clean, documented code ideal for understanding HNSW mechanics
- 📊 **Production-Ready**: Wrapper around battle-tested hnswlib for real-world applications
- 🧠 **Intelligent Adaptation**: Learns from query patterns to optimize search performance
- 📈 **Extensive Analysis**: Built-in benchmarking, visualization, and performance tracking
- 🔬 **Research-Oriented**: Designed for experimentation with adaptation strategies

---

## Key Features

### Core Capabilities

- **Dual Implementation Approach**
  - From-scratch Python implementation for education
  - hnswlib wrapper for production performance
  - Side-by-side comparison and validation

- **Adaptive Optimization Framework**
  - Query pattern logging and analysis
  - Frequency-based adaptation strategy
  - Pluggable adapter architecture for custom strategies

- **Comprehensive Benchmarking**
  - Recall@K, Precision, MAP, nDCG metrics
  - Latency percentiles (P50, P95, P99)
  - QPS (Queries Per Second) measurement
  - Automated visualization generation

- **Standard Dataset Support**
  - SIFT1M, GIST1M benchmarks
  - GloVe, NYTimes datasets
  - Custom dataset integration
  - Automatic download scripts

---

## Theoretical Background

### Hierarchical Navigable Small World (HNSW)

HNSW is a state-of-the-art graph-based algorithm for approximate nearest neighbor search, introduced by Malkov and Yashunin (2018). It builds upon the concept of navigable small world networks to create an efficient multi-layer graph structure.

#### Key Concepts

**1. Hierarchical Structure**
```
Layer 2:  O ---------> O (sparse, long-range connections)
           |           |
Layer 1:  O --> O --> O --> O (medium density)
           |     |     |     |
Layer 0:  O-O-O-O-O-O-O-O-O-O (dense, all nodes present)
```

- **Layer 0**: Contains all data points with dense local connections
- **Upper Layers**: Progressively sparser, enabling fast traversal to target regions
- **Entry Point**: Top-level node for search initiation

**2. Graph Construction Algorithm**

For each new element:
1. **Layer Assignment**: Randomly assign layer using exponential distribution
   ```
   layer = floor(-ln(uniform(0,1)) × mL)
   where mL = 1/ln(M)
   ```

2. **Top-Down Search**: Navigate from entry point to target layer
3. **Neighbor Selection**: Find ef_construction nearest neighbors at each layer
4. **Bidirectional Linking**: Create connections with M nearest neighbors
5. **Pruning**: Maintain M maximum connections per node (2M for layer 0)

**3. Search Algorithm**

```python
def search(query, k, ef):
    # Phase 1: Navigate to layer 0
    current = entry_point
    for layer in range(max_layer, 0, -1):
        current = greedy_search(query, current, 1, layer)
    
    # Phase 2: Detailed search at layer 0
    candidates = beam_search(query, current, ef, layer=0)
    return top_k(candidates, k)
```

**4. Key Parameters**

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|--------|
| **M** | Max connections per node | 12-48 | Higher M → Better recall, more memory |
| **ef_construction** | Candidate list size during build | 100-500 | Higher ef → Better quality, slower build |
| **ef_search** | Candidate list size during search | k-1000 | Higher ef → Better recall, slower search |
| **mL** | Layer multiplier | 1/ln(M) | Controls layer distribution |

**5. Complexity Analysis**

- **Build Time**: O(N × log(N) × M × ef_construction)
- **Search Time**: O(log(N) × M × ef_search)
- **Memory**: O(N × M × avg_layers)

Where N is the number of data points.

### Adaptive HNSW Enhancement

This project extends standard HNSW with adaptive optimization:

**Frequency-Based Adaptation**
- Monitors which nodes are frequently accessed during searches
- Identifies "hub" nodes in the query distribution
- Simulates optimization strategies (e.g., entry point adjustment, connection prioritization)

**Performance Gains Observed**
- Up to **14.37% latency reduction**
- Up to **16.60% throughput improvement**
- **Zero degradation** in recall quality

---

## Project Structure

```
Adaptive-HNSW/
├── src/                          # Source code
│   ├── hnsw_from_scratch.py     # Pure Python HNSW implementation
│   ├── hnsw_wrapper.py          # hnswlib C++ wrapper
│   ├── adaptive_hnsw.py         # Adaptive optimization framework
│   ├── benchmark.py             # Comprehensive benchmarking suite
│   ├── metrics.py               # Evaluation metrics (Recall, MAP, nDCG)
│   ├── visualize_benchmark.py   # Visualization generation
│   ├── data_loader.py           # Dataset loading utilities
│   ├── query_logger.py          # Query pattern logging
│   ├── evaluation.py            # Performance evaluation
│   ├── utils.py                 # Helper functions
│   ├── test_quick.py            # Quick validation tests
│   ├── adaptation/              # Adaptive strategies
│   │   ├── base_adapter.py      # Abstract adapter interface
│   │   └── frequency_adapter.py # Frequency-based optimization
│   └── scripts/
│       ├── main.py              # Main execution script
│       ├── download_datasets.py # Dataset downloader
│       └── data_download-Instruction.md
│
├── data/                        # Datasets
│   ├── sift1m/                 # SIFT1M benchmark
│   ├── gist1m/                 # GIST1M benchmark
│   └── glove100/               # GloVe-100 embeddings
│
├── results/                     # Experiment results
│   ├── indices/                # Saved HNSW indices
│   ├── logs/                   # Query logs
│   ├── metrics/                # Performance metrics (JSON)
│   ├── plots/                  # Generated visualizations
│   └── analysis.md             # Detailed analysis report
│
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── visualize.py               # Standalone visualization script
├── expt.ipynb                # Jupyter notebook experiments
└── readme.md                  # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Conda for environment management

### Step 1: Clone Repository

```bash
git clone https://github.com/Husain-Malwat/Adaptive-HNSW.git
cd Adaptive-HNSW
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n adaptive-hnsw python=3.8
conda activate adaptive-hnsw
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.20.0          # Numerical computations
hnswlib>=0.8.0         # C++ HNSW library
scikit-learn>=1.0.0    # Metrics and utilities
matplotlib>=3.5.0      # Visualization
seaborn>=0.11.0        # Statistical plots
pyyaml>=5.4.0          # Configuration parsing
tqdm>=4.60.0           # Progress bars
pandas>=1.3.0          # Data analysis
```

---

## Quick Start

### 1. Run Quick Validation Test

Verify both implementations are working correctly:

```bash
cd src
python test_quick.py
```

**Expected Output:**
```
============================================================
TESTING HNSW FROM SCRATCH - BASIC FUNCTIONALITY
============================================================
✓ Index built: 1000 elements, max_layer=13
✓ All tests passed!

============================================================
COMPARING SCRATCH vs HNSWLIB - SMALL DATASET
============================================================
✓ Average overlap: 99.0%
✅ ALL TESTS PASSED - Ready for full benchmark!
```

### 2. Download Standard Datasets

```bash
cd src/scripts
python download_datasets.py --datasets sift1m gist1m
```

### 3. Run Adaptive HNSW Experiment

```bash
cd src/scripts
python main.py ../../config.yaml
```

This will:
1. Load the dataset specified in `config.yaml`
2. Build an HNSW index
3. Run baseline benchmarks
4. Execute warmup queries and log patterns
5. Analyze query patterns
6. Apply adaptive optimization
7. Re-benchmark and compare results

### 4. Run Comprehensive Benchmark

Compare from-scratch vs hnswlib implementations:

```bash
cd src
python benchmark.py
```

### 5. Generate Visualizations

```bash
cd src
python visualize_benchmark.py ../results/benchmark_results.json ../results/plots/
```

---

## Benchmarking

### Running Benchmarks

The benchmark suite compares both implementations across multiple dimensions:

```python
from benchmark import BenchmarkRunner

config = {
    'use_dummy_data': True,
    'num_base': 10000,
    'num_queries': 100,
    'dim': 128,
    'k': 10,
    'space': 'l2',
    'M': 16,
    'ef_construction': 200,
    'ef_search_values': [50, 100, 200, 400]
}

runner = BenchmarkRunner(config)
results = runner.run_full_benchmark(force_rebuild=False)
```

### Metrics Computed

1. **Quality Metrics**
   - Recall@K: Fraction of true neighbors found
   - Precision@K: Accuracy of retrieved neighbors
   - MAP@K: Mean Average Precision
   - nDCG@K: Normalized Discounted Cumulative Gain

2. **Performance Metrics**
   - Latency: Mean, Median, P50, P90, P95, P99
   - QPS: Queries per second
   - Build Time: Index construction time
   - Memory Usage: Index size

3. **Comparison Metrics**
   - Result Overlap: Agreement between implementations
   - Speedup Factor: Performance ratio
   - Recall Difference: Quality delta

### Example Output

```
Build Performance:
  Scratch:  125.34s
  HNSWLib:  2.47s
  Build Speedup: 50.74x (hnswlib is faster)

Search Performance (ef_search -> Recall@k, Latency):

  ef_search=100:
    Scratch:  Recall=0.9245, Latency=12.34ms, QPS=81.0
    HNSWLib:  Recall=0.9248, Latency=0.23ms, QPS=4347.8
    Speedup:  53.65x
    Overlap:  99.2%
```

---

## Adaptive Strategies

### Frequency-Based Adaptation

The **FrequencyAdapter** analyzes query logs to identify frequently accessed nodes:

```python
from adaptation.frequency_adapter import FrequencyAdapter

adapter = FrequencyAdapter(config)
adapter.analyze('logs/query_log.jsonl')
insights = adapter.get_adaptation_insights(top_n=10)
```

**Insights Generated:**
- Most frequently accessed node IDs
- Access count distribution
- Hub node identification
- Query pattern clustering potential

### Creating Custom Adapters

Extend the `BaseAdapter` class to implement new strategies:

```python
from adaptation.base_adapter import BaseAdapter

class MyCustomAdapter(BaseAdapter):
    def analyze(self, log_file_path):
        # Implement pattern analysis
        pass
    
    def get_adaptation_insights(self):
        # Return optimization recommendations
        pass
```

**Potential Adaptation Strategies:**

1. **Spatial Clustering**: Group queries by region in vector space
2. **Temporal Patterns**: Identify time-based query shifts
3. **Path Optimization**: Analyze traversal paths for shortcuts
4. **Dynamic Entry Points**: Adjust entry points based on query distribution
5. **Connection Reweighting**: Prioritize edges on frequent paths

---

## Performance Results

### GIST1M Dataset (960-dimensional vectors)

| Metric | EF=50 Baseline | EF=50 Adapted | Improvement |
|--------|----------------|---------------|-------------|
| **Latency** | 0.96 ms | 0.85 ms | **↓ 11.04%** |
| **QPS** | 1019.6 | 1127.6 | **↑ 10.60%** |
| **Recall@10** | 0.8257 | 0.8257 | No degradation |

| Metric | EF=100 Baseline | EF=100 Adapted | Improvement |
|--------|-----------------|----------------|-------------|
| **Latency** | 1.02 ms | 0.88 ms | **↓ 14.37%** |
| **QPS** | 973.8 | 1135.4 | **↑ 16.60%** |
| **Recall@10** | 0.8257 | 0.8257 | No degradation |

### Implementation Comparison (10K vectors, 128-dim)

| Implementation | Build Time | Query Latency | Recall@10 | QPS |
|----------------|------------|---------------|-----------|-----|
| **Custom** | 125.3s | 12.34ms | 92.45% | 81.0 |
| **HNSWLib** | 2.5s | 0.23ms | 92.48% | 4347.8 |
| **Speedup** | 50.7x | 53.7x | ≈ same | 53.7x |

**Key Findings:**
- Custom implementation (From-scratch) achieves **99%+ result overlap** with hnswlib
- Adaptive optimization provides **10-16% performance boost**
- Zero recall degradation after adaptation

---

## Datasets

### Supported Datasets

1. **SIFT1M** (128-dimensional SIFT descriptors)
   - 1M base vectors
   - 10K query vectors
   - 100 ground truth neighbors per query

2. **GIST1M** (960-dimensional GIST descriptors)
   - 1M base vectors
   - 1K query vectors
   - 100 ground truth neighbors per query

3. **GloVe-100** (100-dimensional word embeddings)
   - Custom size (configurable)
   - Cosine similarity metric

4. **NYTimes** (256-dimensional article embeddings)
   - Custom size (configurable)

### Dataset Format

- **Base vectors**: `.fvecs` format (float32 vectors)
- **Query vectors**: `.fvecs` format
- **Ground truth**: `.ivecs` format (int32 neighbor IDs)

### Download Instructions

```bash
cd src/scripts
python download_datasets.py --datasets sift1m gist1m --data-dir ../../data/
```

**See** `src/scripts/data_download-Instruction.md` **for detailed instructions.**

---

## Configuration

### Parameter Tuning Guidelines

**M (connections per node)**
- Small (4-8): Fast build, lower recall
- Medium (12-24): Balanced performance
- Large (32-64): High recall, more memory

**ef_construction**
- Low (50-100): Fast build, may reduce quality
- Medium (100-300): Standard choice
- High (400-1000): Maximum quality, slow build

**ef_search**
- Must be ≥ k (number of neighbors)
- Higher values: Better recall, slower queries
- Tune based on recall/latency tradeoff

---

## Architecture

<!-- ### Class Hierarchy

```
BaseAdapter (ABC)
    ├── FrequencyAdapter
    └── [Custom Adapters]

HNSWScratch
    └── Pure Python HNSW implementation

HNSWWrapper
    └── hnswlib C++ binding

AdaptiveHNSW
    ├── HNSWWrapper
    ├── QueryLogger
    └── List[BaseAdapter]

BenchmarkRunner
    ├── HNSWScratch
    └── HNSWWrapper

MetricsCalculator
    └── Static metric computation methods

BenchmarkVisualizer
    └── Plotting and chart generation
``` -->

### Data Flow

```
[Dataset] → [DataLoader] → [Base Vectors]
                                    ↓
                          [AdaptiveHNSW.build()]
                                    ↓
                            [HNSW Index]
                                    ↓
[Queries] → [AdaptiveHNSW.search_and_log()] → [Results + Logs]
                                    ↓
                         [FrequencyAdapter.analyze()]
                                    ↓
                          [Adaptation Insights]
                                    ↓
                    [Apply Optimization (Simulated)]
                                    ↓
                        [Re-benchmark & Compare]
```

---

## Contributing

Contributions are welcome! Here's how you can help:

### Areas for Contribution

1. **New Adaptation Strategies**
   - Spatial clustering adapters
   - Temporal pattern analysis
   - Path optimization algorithms

2. **Performance Improvements**
   - Cython/Numba optimization for from-scratch implementation
   - Parallel query processing
   - Memory optimization

3. **Additional Metrics**
   - Distance ratio metrics
   - Index quality measures
   - Resource utilization tracking

4. **Dataset Support**
   - More standard benchmarks
   - Custom data format loaders
   - Synthetic data generators

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/Adaptive-HNSW.git
cd Adaptive-HNSW

# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and test
python src/test_quick.py
python src/benchmark.py

# Commit and push
git commit -am "Add my new feature"
git push origin feature/my-new-feature

# Create Pull Request on GitHub
```


## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{adaptive_hnsw_2025,
  author = {Husain Malwat, Balgopal Moharana, Rohit Raj, Anirban Dasgupta},
  title = {Adaptive HNSW: Intelligent Vector Search with Query-Driven Optimization},
  year = {2025},
  url = {https://github.com/Husain-Malwat/Adaptive-HNSW}
}
```

### References

**Original HNSW Paper:**
```bibtex
@article{malkov2018efficient,
  title={Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs},
  author={Malkov, Yu A and Yashunin, Dmitry A},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={42},
  number={4},
  pages={824--836},
  year={2018},
  publisher={IEEE}
}
```

---

## 🙏 Acknowledgments

- **Prof. Anirban Dasgupta, IIT Gandhinagar**: For invaluable guidance and mentorship throughout this research project
- **hnswlib**: Fast C++ implementation by @nmslib
- **SIFT/GIST datasets**: Provided by INRIA/TEXMEX
- **Research Community**: For advancing ANN search algorithms

---



<div align="center">

**⭐ Star this repository if you find it helpful!**


</div>