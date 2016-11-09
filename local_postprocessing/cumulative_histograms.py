from __future__ import division, print_function
import os
from glob import glob
import zlib
import io
import numpy as np
import click
import tqdm

from healthybrains.inputoutput import id_from_file_name


@click.command()
@click.argument("folder", nargs=1)
@click.argument("output", nargs=1)
@click.option("--zoom", type=int, default=1)
def main(folder, output, zoom):
    max_l = 20 * zoom
    file_names = glob(os.path.join(folder, "*"))
    wbits = zlib.MAX_WBITS | 16
    arrays = np.zeros(shape=(len(file_names), max_l - 2))
    for file_name in tqdm.tqdm(file_names):
        decompressor = zlib.decompressobj(wbits)
        with open(file_name, "rb") as infile:
            stringio = io.BytesIO()
            decompressed = decompressor.decompress(infile.read())
            stringio.write(decompressed)
            stringio.seek(0)
            name = id_from_file_name(np.load(stringio)[0])
            thicknesses = np.load(stringio).flatten()
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
