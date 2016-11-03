import numpy as np
import click
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


@click.command()
@click.argument("health", nargs=1)
@click.argument("thickness_folder", nargs=1)
def main(health, thickness_folder):
    # dfs = []
    # health_id = np.genfromtxt(health)
    # for file_name in glob.glob(os.path.join(thickness_folder, "*")):
        # index = int(os.path.splitext(os.path.basename(file_name))[0].split("_")[1]) - 1
        # dfs.append(
            # pd.DataFrame({
                # "id": index,
                # "thickness": np.load(file_name),
                # "health": health_id[index]}))
    # df = pd.concat(dfs)
    x = np.arange(100)
    y = np.arange(100)
    plt.figure()
    plt.scatter(x, y)



if __name__ == "__main__":
    main()
