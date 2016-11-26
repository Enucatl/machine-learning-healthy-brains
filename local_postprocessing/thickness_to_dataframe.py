import click
import os
import numpy as np
import pandas as pd
import zlib
import io
from tqdm import tqdm

from healthybrains.inputoutput import id_from_file_name

@click.command()
@click.argument("input_files", nargs=-1)
@click.option("--targets", nargs=1, default="../data/targets.csv")
def main(input_files, targets):
    healths = np.genfromtxt(targets, dtype=np.uint8)
    fr = np.zeros_like(healths)
    cen = np.zeros_like(healths)
    cer = np.zeros_like(healths)
    for file_name in tqdm(input_files):
        wbits = zlib.MAX_WBITS | 16
        decompressor = zlib.decompressobj(wbits)
        infile = open(file_name, "rb")
        stringio = io.BytesIO()
        decompressed = decompressor.decompress(infile.read())
        infile.close()
        stringio.write(decompressed)
        stringio.seek(0)
        name = np.load(stringio)[0]
        file_id = id_from_file_name(name)
        health = healths[file_id - 1]
        cortex = np.load(stringio)
        frontal = cortex[25:150, 70:125, 75:150]
        central = cortex[40:130, 145:190, 45:130]
        cerebellum = cortex[40:130, 35:90, 18:40]
        fr[file_id - 1] = np.median(frontal[frontal > 0])
        cen[file_id - 1] = np.median(central[central > 0])
        cer[file_id - 1] = np.median(cerebellum[cerebellum > 0])
    df = pd.DataFrame(data={
        "health": healths,
        "frontal": fr,
        "central": cen,
        "cerebellum": cer})
    output_name = os.path.dirname(input_files[0]) + "_medians.csv"
    df.to_csv(output_name)


if __name__ == "__main__":
    main()
