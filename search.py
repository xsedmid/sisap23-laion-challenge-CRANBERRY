import argparse
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time 

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def prepare(kind, size):
    url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    task = {
        "query": f"{url}/public-queries-10k-{kind}.h5",
        "dataset": f"{url}/laion2B-en-{kind}-n={size}.h5",
    }

    for version, url in task.items():
        download(url, os.path.join("data", kind, size, f"{version}.h5"))

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

def run(kind, key, size="100K", k=30):
    print("Running", kind)
    
    prepare(kind, size)

    data = np.array(h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")[key])
    queries = np.array(h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key])
    print(f'data.shape={data.shape}, queries.shape={queries.shape}')

    print(f"*** Running Java-based implementation (building the index + searching)...")
    start = time.time()


    elapsed_build = time.time() - start
    print(f"*** Done in {elapsed_build}s.")
    

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

    # run("pca32v2", "pca32", args.size, args.k)
    run("pca96v2", "pca96", args.size, args.k)
    # run("hammingv2", "hamming", args.size, args.k)
