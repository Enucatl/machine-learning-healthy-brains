from __future__ import division, print_function
import click
import numpy as np
from sklearn.metrics import pairwise_distances
import csv


@click.command()
@click.argument("train_fc")
@click.argument("train_fc_groups")
@click.argument("test_fc")
def main(train_fc, train_fc_groups, test_fc):
    train_fc = np.load(train_fc)
    train_fc_groups = np.genfromtxt(
        train_fc_groups,
        skip_header=1,
        dtype=np.uint8) - 1
    test_fc = np.load(test_fc)
    train_health = np.genfromtxt(
        "../data/targets.csv",
        dtype=np.uint8)
    fc_group_meaning = np.zeros(2, dtype=np.float)
    for i in range(2):
        selected_healths = train_health[train_fc_groups == i]
        fc_group_meaning[i] = np.sum(selected_healths) / np.size(selected_healths)
    print(fc_group_meaning)
    healthy = np.argmax(fc_group_meaning)
    diseased = np.argmin(fc_group_meaning)
    status = [diseased, healthy]
    diagnosis = []
    for test_frontal_central in test_fc:
        d_from_diseased = np.max(pairwise_distances(
            train_fc[train_fc_groups == diseased],
            test_frontal_central.reshape(1, -1),
            metric="chebyshev"
        ))
        d_from_healthy = np.max(pairwise_distances(
            train_fc[train_fc_groups == healthy],
            test_frontal_central.reshape(1, -1),
            metric="chebyshev"
        ))
        distances = np.array((
            d_from_diseased,
            d_from_healthy,
            ))
        min_dist = np.argmin(distances)
        diagnosis.append(fc_group_meaning[status[min_dist]])

    with open("prediction.csv", "w") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["ID", "Prediction"])
        for i, d in enumerate(diagnosis):
            writer.writerow([i + 1, d])


if __name__ == "__main__":
    main()
