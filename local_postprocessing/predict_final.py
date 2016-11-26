from __future__ import division, print_function
import click
import numpy as np
from sklearn.metrics import pairwise_distances
import csv


@click.command()
@click.argument("train_fc")
@click.argument("train_c")
@click.argument("train_fc_groups")
@click.argument("train_c_groups")
@click.argument("test_fc")
@click.argument("test_c")
def main(train_fc, train_c, train_fc_groups, train_c_groups, test_fc, test_c):
    train_fc = np.load(train_fc)
    train_c = np.load(train_c)
    train_fc_groups = np.genfromtxt(
        train_fc_groups,
        skip_header=1,
        dtype=np.uint8) - 1
    train_c_groups = np.genfromtxt(
        train_c_groups,
        skip_header=1,
        dtype=np.uint8) - 1
    test_fc = np.load(test_fc)
    test_c = np.load(test_c)
    train_health = np.genfromtxt(
        "../data/targets.csv",
        dtype=np.uint8)
    fc_group_meaning = np.zeros(3, dtype=np.float)
    for i in range(3):
        selected_healths = train_health[train_fc_groups == i]
        fc_group_meaning[i] = np.sum(selected_healths) / np.size(selected_healths)
    print(fc_group_meaning)
    healthy = np.argmax(fc_group_meaning)
    diseased = np.argmin(fc_group_meaning)
    dubious = np.where(np.logical_and(
        fc_group_meaning != np.max(fc_group_meaning),
        fc_group_meaning != np.min(fc_group_meaning)))[0][0]
    status = (healthy, diseased, dubious)
    c_group_meaning = np.zeros(3, dtype=np.float)
    for i in range(3):
        selected_healths = train_health[
            np.logical_and(
                train_c_groups == i,
                train_fc_groups == dubious)]
        c_group_meaning[i] = np.sum(selected_healths) / np.size(selected_healths)
    c_healthy = np.argmax(c_group_meaning)
    c_diseased = np.argmin(c_group_meaning)
    c_dubious = np.where(np.logical_and(
        c_group_meaning != np.max(c_group_meaning),
        c_group_meaning != np.min(c_group_meaning)))[0][0]
    c_status = (c_healthy, c_diseased, c_dubious)
    print(c_group_meaning)
    diagnosis = []
    for test_frontal_central, test_cerebellum in zip(test_fc, test_c):
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
        d_from_dubious = np.max(pairwise_distances(
            train_fc[train_fc_groups == dubious],
            test_frontal_central.reshape(1, -1),
            metric="chebyshev"
        ))
        distances = np.array((
            d_from_healthy,
            d_from_diseased,
            d_from_dubious))
        min_dist = np.argmin(distances)
        if min_dist == dubious:
            d_from_diseased = np.max(pairwise_distances(
                train_c[train_c_groups == c_diseased],
                test_cerebellum.reshape(1, -1),
                metric="chebyshev"
            ))
            d_from_healthy = np.max(pairwise_distances(
                train_c[train_c_groups == c_healthy],
                test_cerebellum.reshape(1, -1),
                metric="chebyshev"
            ))
            d_from_dubious = np.max(pairwise_distances(
                train_c[train_c_groups == c_dubious],
                test_cerebellum.reshape(1, -1),
                metric="chebyshev"
            ))
            distances = np.array((
                d_from_diseased,
                d_from_healthy,
                d_from_dubious))
            min_dist = np.argmin(distances)
            diagnosis.append(c_group_meaning[c_status[min_dist]])
        else:
            diagnosis.append(fc_group_meaning[status[min_dist]])

    with open("prediction.csv", "w") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["ID", "Prediction"])
        for i, d in enumerate(diagnosis):
            writer.writerow([i + 1, d])



if __name__ == "__main__":
    main()
