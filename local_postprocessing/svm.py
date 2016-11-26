import click
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@click.command()
@click.argument("input_file", nargs=1)
def main(input_file):
    df = pd.DataFrame.from_csv(input_file)
    plt.figure()
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_upper(plt.scatter)
    g.map_diag(sns.kdeplot, lw=3)
    plt.show()
    plt.ion()
    input()


if __name__ == "__main__":
    main()
