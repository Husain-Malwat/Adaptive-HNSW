# Dataset Download Instructions

This guide provides instructions for downloading the SIFT1M and GIST1M datasets using the `download_datasets.py` script.

## Prerequisites

Ensure you have the necessary Python libraries installed. The `tensorflow` and `tensorflow-datasets` packages are only required for the recommended SIFT1M download method.

## Method 1: Using the Download Script (Recommended)

The `download_datasets.py` script automates the download and extraction process, placing files in the correct `data/` subdirectories.

### For SIFT1M (Highly Recommended: Use TensorFlow)

The original FTP server for the datasets is often slow or unavailable. Using TensorFlow Datasets is a much more reliable method.

1.  Navigate to the project's root directory.
2.  Run the script with the `--use_tensorflow` flag:

    ```bash
    python src/utils/download_datasets.py --dataset sift1m --use_tensorflow
    ```

3.  The script will download the data, process it into the required `.fvecs` and `.ivecs` formats, and place it in the `data/sift1m/` directory. **Note:** The `learn` component is not included in the TensorFlow version, which is acceptable for this project.

### For GIST1M (or SIFT1M via FTP)

This method relies on the original FTP servers. The script will download the single large archive for each dataset, extract the components, and then clean up.

1.  Navigate to the project's root directory.
2.  Run the script for the desired dataset:

    ```bash
    # For GIST1M
    python src/utils/download_datasets.py --dataset gist1m

    # For SIFT1M (if the TensorFlow method fails for any reason)
    python src/utils/download_datasets.py --dataset sift1m
    ```

3.  The script will download, extract, and place the files in `data/gist1m/` or `data/sift1m/`.

## Method 2: Manual Download (Fallback Plan)

If the script fails, you can download and extract the files manually.

### For SIFT1M

1.  **Download**: Get the single archive `sift.tar.gz` from `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`.
2.  **Create Directory**: `mkdir -p data/sift1m`
3.  **Move & Extract**: Move `sift.tar.gz` into `data/sift1m/` and run these commands from within that directory:
    ```bash
    # 1. Extract the archive
    tar -xzvf sift.tar.gz
    
    # 2. Move and rename files from the 'sift' subfolder
    mv sift/sift_base.fvecs sift1m_base.fvecs
    mv sift/sift_query.fvecs sift1m_query.fvecs
    mv sift/sift_groundtruth.ivecs sift1m_groundtruth.ivecs
    mv sift/sift_learn.fvecs sift1m_learn.fvecs
    
    # 3. Clean up
    rm -r sift
    rm sift.tar.gz
    ```
4.  **Verify**: Your `data/sift1m` directory should now contain the four renamed `.fvecs`/`.ivecs` files.

### For GIST1M

1.  **Download**: Get the single archive `gist.tar.gz` from `ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz`.
2.  **Create Directory**: `mkdir -p data/gist1m`
3.  **Move & Extract**: Move `gist.tar.gz` into `data/gist1m/` and run these commands from within that directory:
    ```bash
    # 1. Extract the archive
    tar -xzvf gist.tar.gz

    # 2. Move and rename files
    mv gist/gist_base.fvecs gist1m_base.fvecs
    mv gist/gist_query.fvecs gist1m_query.fvecs
    mv gist/gist_groundtruth.ivecs gist1m_groundtruth.ivecs

    # 3. Clean up
    rm -r gist
    rm gist.tar.gz
    ```
4.  **Verify**: Your `data/gist1m` directory should now contain the three renamed `.fvecs`/`.ivecs` files.