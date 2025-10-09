import os
import argparse
import requests
import gzip
import shutil
import tarfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from urllib.parse import urlparse
import urllib.request
import urllib.error

# --- Dataset Definitions ---
# Corrected: Note that all components are in a single tar.gz archive.
DATASET_URLS = {
    "sift1m": {
        "base": {
            "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
            "archive_name": "sift.tar.gz",
            "final_name": "sift1m_base.fvecs",
            "file_in_archive": "sift/sift_base.fvecs"
        },
        "query": {
            "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
            "archive_name": "sift.tar.gz",
            "final_name": "sift1m_query.fvecs",
            "file_in_archive": "sift/sift_query.fvecs"
        },
        "ground_truth": {
            "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
            "archive_name": "sift.tar.gz",
            "final_name": "sift1m_groundtruth.ivecs",
            "file_in_archive": "sift/sift_groundtruth.ivecs"
        },
        "learn": {
            "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
            "archive_name": "sift.tar.gz",
            "final_name": "sift1m_learn.fvecs",
            "file_in_archive": "sift/sift_learn.fvecs"
        }
    },
    "gist1m": {
        "base": {
            "url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
            "archive_name": "gist.tar.gz",
            "final_name": "gist1m_base.fvecs",
            "file_in_archive": "gist/gist_base.fvecs"
        },
        "query": {
            "url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
            "archive_name": "gist.tar.gz",
            "final_name": "gist1m_query.fvecs",
            "file_in_archive": "gist/gist_query.fvecs"
        },
        "ground_truth": {
            "url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
            "archive_name": "gist.tar.gz",
            "final_name": "gist1m_groundtruth.ivecs",
            "file_in_archive": "gist/gist_groundtruth.ivecs"
        }
    }
}


class TqdmUpTo(tqdm):
    """Provides `update_to(block_num, block_size, total_size)` hook for urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, destination):
    """Downloads a file from a URL, handling HTTP, HTTPS, and FTP with a progress bar."""
    parsed_url = urlparse(url)
    filename = os.path.basename(destination)
    print(f"Downloading {filename} from {url}...")

    if parsed_url.scheme == 'ftp':
        try:
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
            return True
        except (urllib.error.URLError, Exception) as e:
            print(f"Error downloading {url} via FTP: {e}")
            if os.path.exists(destination): os.remove(destination)
            return False
    elif parsed_url.scheme in ('http', 'https'):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(destination, 'wb') as f, tqdm(desc=filename, total=total_size, unit='iB', unit_scale=True) as bar:
                for data in response.iter_content(chunk_size=1024):
                    bar.update(len(data))
                    f.write(data)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url} via HTTP/S: {e}")
            if os.path.exists(destination): os.remove(destination)
            return False
    else:
        print(f"Error: Unsupported URL scheme '{parsed_url.scheme}'")
        return False

def extract_from_tar(tar_path, member_name, output_path):
    """Extracts a specific member file from a .tar or .tar.gz archive."""
    print(f"Extracting '{member_name}' from {os.path.basename(tar_path)}...")
    mode = "r:gz" if tar_path.endswith(".tar.gz") else "r"
    try:
        with tarfile.open(tar_path, mode) as tar:
            member = tar.getmember(member_name)
            f_in = tar.extractfile(member)
            if f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return True
            return False
    except (KeyError, tarfile.TarError, Exception) as e:
        print(f"Error extracting {member_name} from {tar_path}: {e}")
        return False

def save_as_fvecs(embeddings, output_path):
    """Saves a numpy array in the fvecs format."""
    print(f"Saving {embeddings.shape[0]} vectors to {output_path}...")
    with open(output_path, 'wb') as f:
        for vec in embeddings:
            dim = np.array([vec.shape[0]], dtype=np.int32)
            f.write(dim.tobytes())
            f.write(vec.astype(np.float32).tobytes())

def save_as_ivecs(indices, output_path):
    """Saves a numpy array in the ivecs format."""
    print(f"Saving {indices.shape[0]} vectors to {output_path}...")
    with open(output_path, 'wb') as f:
        for idx_row in indices:
            dim = np.array([idx_row.shape[0]], dtype=np.int32)
            f.write(dim.tobytes())
            f.write(idx_row.astype(np.int32).tobytes())

def main():
    parser = argparse.ArgumentParser(description="Download and prepare common ANN benchmark datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_URLS.keys()) + ["all"], help="Dataset to download.")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory for datasets.")
    parser.add_argument("--force", action="store_true", help="Force re-download and extraction.")
    parser.add_argument("--no_cleanup", action="store_false", dest="cleanup", help="Do not remove archive files after extraction.")
    parser.add_argument("--use_tensorflow", action="store_true", help="Use TensorFlow Datasets for SIFT1M (recommended).")
    
    args = parser.parse_args()
    
    datasets_to_process = list(DATASET_URLS.keys()) if args.dataset == "all" else [args.dataset]
    
    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Target data directory: {args.data_dir}")
    print(f"Datasets to process: {', '.join(datasets_to_process)}\n" + "-"*30)
    
    for ds_name in datasets_to_process:
        print(f"Processing dataset: {ds_name}")
        target_ds_dir = os.path.join(args.data_dir, ds_name)
        os.makedirs(target_ds_dir, exist_ok=True)

        # # Method 1: TensorFlow (recommended for sift1m)
        # if ds_name == "sift1m" and args.use_tensorflow:
        #     if download_sift1m_tensorflow(target_ds_dir):
        #         print(f"Successfully processed {ds_name} using TensorFlow.")
        #     else:
        #         print(f"TensorFlow method failed. Try running without '--use_tensorflow' to use FTP.")
        #     print("-" * 30)
        #     continue
            
        # Method 2: FTP (for gist1m or as a fallback for sift1m)
        ds_info = DATASET_URLS[ds_name]
        
        # Download all unique archives for the dataset
        archives_to_download = {info['archive_name']: info['url'] for info in ds_info.values()}
        downloaded_archives = []
        for archive_name, url in archives_to_download.items():
            archive_path = os.path.join(target_ds_dir, archive_name)
            if not os.path.exists(archive_path) or args.force:
                if not download_file(url, archive_path):
                    print(f"FATAL: Could not download required archive {archive_name}. Skipping dataset {ds_name}.")
                    continue # Skip to next dataset
            downloaded_archives.append(archive_path)

        # Extract required components from the archives
        for component_name, component_info in ds_info.items():
            final_file_path = os.path.join(target_ds_dir, component_info["final_name"])
            
            if os.path.exists(final_file_path) and not args.force:
                print(f"  Component '{component_name}' exists. Skipping extraction.")
                continue

            archive_path = os.path.join(target_ds_dir, component_info["archive_name"])
            if not os.path.exists(archive_path):
                 print(f"  Error: Archive {component_info['archive_name']} not found for component {component_name}.")
                 continue
            
            if not extract_from_tar(archive_path, component_info["file_in_archive"], final_file_path):
                 print(f"  Failed to extract {component_name}.")

        if args.cleanup:
            for archive_path in downloaded_archives:
                print(f"Cleaning up {os.path.basename(archive_path)}...")
                os.remove(archive_path)
        
        print(f"Finished processing {ds_name}.")
        print("-" * 30)
    
    print("All specified datasets processed.")

if __name__ == "__main__":
    main()