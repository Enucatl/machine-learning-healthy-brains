from __future__ import division, print_function
import click
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score


@click.command()
@click.argument("input_file")
@click.option("--ground_truth_file", default="../data/targets.csv")
def main(input_file, ground_truth_file):
    arrays = np.load(input_file)
    labels_true = np.genfromtxt(ground_truth_file)
    print(arrays.shape)
    distance = pairwise_distances(arrays, metric="chebyshev")
    print(distance.shape)
    model = AgglomerativeClustering(
        n_clusters=2,
        linkage="complete",
        affinity="precomputed")
    model.fit(distance)
    labels_pred = model.labels_
    print(labels_pred, labels_true)
    print(adjusted_rand_score(labels_pred, labels_true))


if __name__ == "__main__":
    main()
