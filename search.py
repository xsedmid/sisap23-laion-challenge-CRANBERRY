import argparse
import h5py
import numpy as np
import os
from pathlib import Path
import subprocess
from urllib.request import urlretrieve
import time 

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def prepare(root_data_folder, kind, size):
    dataset_url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    data_file_dict = {
        "dataset_orig": f"laion2B-en-clip768v2-n={size}.h5",
        "dataset": f"laion2B-en-{kind}-n={size}.h5",
        "query_orig": f"public-queries-10k-clip768v2.h5",
        "query": f"public-queries-10k-{kind}.h5"
    }

    for version, file_name in data_file_dict.items():
        result_file_path = os.path.join(root_data_folder, file_name)
        if version.startswith("public-q") or (not os.path.exists(result_file_path)):
            download(f"{dataset_url}/{file_name}", result_file_path)

    return data_file_dict

def store_results(dst, algo, kind, D, I, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()

def run(root_data_folder, kind, key, size="100K", k=30):
    print("Running", kind)
    
    data_file_dict = prepare(root_data_folder, kind, size)

    # Download pivots
    pivot_file = "laion2B-en-clip768v2-n=100M.h5_2048pivots.gz"
    pivots_url = f"https://www.fi.muni.cz/~xsedmid/temp/{pivot_file}"
    pivot_dir = os.path.join(root_data_folder, 'Dataset', 'Pivot')

    # Create pivot directory if it does not exist
    if not os.path.exists(pivot_dir):
        os.makedirs(pivot_dir, exist_ok=True)
    download(pivots_url, os.path.join(pivot_dir, pivot_file))


    #data = np.array(h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")[key])
    #queries = np.array(h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key])
    #print(f'data.shape={data.shape}, queries.shape={queries.shape}')

    # root_folder-similarity_search; h5 path: data768, data96pca, query768, query96pca

    dataset_orig = os.path.join(root_data_folder, data_file_dict['dataset_orig'])
    dataset = os.path.join(root_data_folder, data_file_dict['dataset'])
    query_orig = os.path.join(root_data_folder, data_file_dict['query_orig'])
    query = os.path.join(root_data_folder, data_file_dict['query'])

    print(f"*** Running Java-based implementation (building the index + searching)...")
    print(f"*** args:")
    #print(f"  {root_data_folder}")
    print(f"  {dataset_orig}")
    print(f"  {dataset}")
    print(f"  {query_orig}")
    print(f"  {query}")

    print(f'Current path: {os.getcwd()}')
    # Print all files in the current directory
    print(f'Files in current directory: {os.listdir()}')
    #print(f'VMTrials path: {os.path.join(os.getcwd(), "VMTrials").listdir()}')
    # class_files = os.path.join(os.getcwd(), 'VMTrials', 'target', 'classes', 'vm').listdir()
    # print(class_files)
    # class_files = os.path.join(os.getcwd(), 'VMTrials', 'target', 'classes', 'metricSpace').listdir()
    # print(class_files)

    start = time.time()
    #subprocess.check_output(['java', '-cp', 'VMTrials', 'vm.vmtrials.tripleFiltering_Challenge.Main', root_data_folder, dataset_orig, dataset, query_orig, query], universal_newlines=True)
    #subprocess.check_output(['java', '-cp', os.path.join(os.getcwd(), 'VMTrials', 'target', 'classes'), 'vm.vmtrials.tripleFiltering_Challenge.Main', dataset_orig, dataset, query_orig, query, '1'], universal_newlines=True)
    subprocess.check_output(['java', '-jar', os.path.join(os.getcwd(), 'VMTrials', 'target', 'VMTrials-1.0-SNAPSHOT-jar-with-dependencies.jar'), 'vm.vmtrials.tripleFiltering_Challenge.Main', dataset_orig, dataset, query_orig, query, '1'], universal_newlines=True)

    elapsed_build = time.time() - start
    print(f"*** Done in {elapsed_build}s.")
    
    # conversion of .csv results to .h5 format



    # for nprobe in [1, 2, 5, 10, 20, 50, 100]:
    #     print(f"Starting search on {queries.shape} with nprobe={nprobe}")
    #     start = time.time()
    #     index.nprobe = nprobe
    #     D, I = index.search(queries, k)
    #     elapsed_search = time.time() - start
    #     print(f"Done searching in {elapsed_search}s.")

    #     I = I + 1 # FAISS is 0-indexed, groundtruth is 1-indexed

    #     identifier = f"index=({index_identifier}),query=(nprobe={nprobe})"

    # store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), "faissIVF", kind, D, I, elapsed_build, elapsed_search, identifier, size)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        default="100K"
    )
    parser.add_argument(
        "--k",
        default=30,
    )

    args = parser.parse_args()

    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    root_data_folder = "Similarity_search"
    run(root_data_folder, "pca96v2", "pca96", args.size, args.k)
