from __future__ import division, print_function
import click
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import log_loss
from sklearn import svm
from sklearn.model_selection import KFold


@click.command()
@click.argument("output_file", nargs=1)
@click.argument("input_files", nargs=-1)
def main(output_file, input_files):
    arrays = np.zeros((len(input_files), 11), dtype=np.float)
    for input_file in input_files:
        basename = os.path.basename(input_file).split("_")
        file_id = int(basename[2])
        array = np.load(input_file)
        arrays[file_id - 1, :] = array
    np.save(output_file, arrays)


if __name__ == "__main__":
    main()
