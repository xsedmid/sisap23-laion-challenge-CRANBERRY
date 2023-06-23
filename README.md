# CRANBERRY algorithm participating in SISAP 23 LAION2B Challenge

This repository contains the sources and evaluation scripts of the **CRANBERRY** (sear**C**hing with the vo**R**onoi p**A**rtitio**N**ing, **B**inary sk**E**tches and **R**elational simila**R**it**Y**) algorithm participating in the [SISAP 2023 LAION2B Challenge](https://sisap-challenges.github.io/).

## Evaluation

The evaluation of the algorithm is implemented using GitHub Actions (GHA). The provided [CI integration](https://github.com/xsedmid/test-Python2Java/blob/master/.github/workflows/ci.yml) installs requirements (including Java 17), downloads a specified dataset, builds an index on the specified dataset, executes *k*NN queries, and evaluates the results of *k*NN queries using the provided [evaluation script](https://github.com/sisap-challenges/sisap23-laion-challenge-evaluation). The provided evaluation script generates the `res.csv` file which primarily contains the calculated metrics of total index construction time, total query execution time, and average recall (the content of this file is printed on the standard output by the last GHA).

The dataset to be downloaded and indexed is specified by the `--size` parameter (possible values: `"100K"`, `"300K"`, `"10M"`, `"30M"`, `"100M"`, default=`"100K"`). The specified dataset file is downloaded from the [Challenge data site](https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge). The query file `public-queries-10k-clip768v2.h5` to be evaluated over the built index is downloaded from the same site; in case, a different query file with should be evaluated, its name has to be specified by changing the `query_orig` and `query` variables in the `search.py` script:
```
def prepare(root_data_folder, kind, size):
    dataset_url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    data_file_dict = {
        "dataset_orig": [os.path.join(root_data_folder, 'Dataset', 'Dataset'), f"laion2B-en-clip768v2-n={size}.h5"],
        "dataset": [os.path.join(root_data_folder, 'Dataset', 'Dataset'), f"laion2B-en-{kind}-n={size}.h5"],
        "query_orig": [os.path.join(root_data_folder, 'Dataset', 'Query'), f"public-queries-10k-clip768v2.h5"],
        "query": [os.path.join(root_data_folder, 'Dataset', 'Query'), f"public-queries-10k-{kind}.h5"]
    }
```
The `dataset_orig` and `query_orig` represent 768-dimensional features, while the `dataset` and `query` stand for PCA96 features.

## Description of the CRANBERRY algorithm

CRANBERRY is a three-phase similarity-search approach implemented in Java (verified on Java JDK 17). Individual phases:
- (1) Looking for the most relevant cells of Voronoi-like partitioning (stops when 0.5M objects are obtained in the identified cells in total) with respect to a query (PCA96 representation);
- (2) Filtering the candidate set of 0.5M objects using: sketches, secondary filtering, and the SimREL technique, with respect to the query;
- (3) Calculating the distances between the query and each retained candidate from the previous step (based on the original 768D represenatations) and returning the *k* most similar as the query result.
