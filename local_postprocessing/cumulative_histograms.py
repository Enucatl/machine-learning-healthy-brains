from __future__ import division, print_function
import os
from glob import glob
import zlib
import io
import numpy as np
import click
from tqdm import tqdm
import pandas as pd

from healthybrains.inputoutput import id_from_file_name


@click.command()
@click.argument("file_names", nargs=-1)
def main(file_names, output):
    max_l = 20
    wbits = zlib.MAX_WBITS | 16
    arrays = np.zeros(shape=(len(file_names), max_l - 2))
    for file_name in tqdm(file_names):
        decompressor = zlib.decompressobj(wbits)
        with open(file_name, "rb") as infile:
            stringio = io.BytesIO()
            decompressed = decompressor.decompress(infile.read())
            stringio.write(decompressed)
            stringio.seek(0)
            name = id_from_file_name(np.load(stringio)[0])
            cortex = np.load(stringio).flatten()
            frontal = cortex[40:130, 140:190, 45:160]
            central = cortex[25:150, 70:125, 75:150]
            cerebellum = cortex[40:130, 35:90, 15:40]
            hist, _ = np.histogram(
                thicknesses[
                    np.logical_and(
                        thicknesses > 0,
                        thicknesses <= max_l)],
                np.arange(1, max_l),
                density=True)
            cumulative = np.cumsum(hist)
            arrays[name - 1, :] = cumulative
    np.save(output, arrays)


if __name__ == "__main__":
    main()
