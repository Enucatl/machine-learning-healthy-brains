import numpy as np
import click
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob


@click.command()
@click.argument("health", nargs=1)
@click.argument("thickness_folder", nargs=1)
def main(health, thickness_folder):
    dfs = []
    health_id = np.genfromtxt(health)
    for file_name in tqdm(glob(os.path.join(thickness_folder, "*"))):
        index = int(os.path.splitext(os.path.basename(file_name))[0].split("_")[1]) - 1
        dfs.append(
            pd.DataFrame({
                "id": index,
                "thickness": np.load(file_name)[..., 3],
                "health": health_id[index]}))
    df = pd.concat(dfs)
    df.to_csv("../data/cortical_thickness.csv.gz", compression="gzip")


if __name__ == "__main__":
    main()
