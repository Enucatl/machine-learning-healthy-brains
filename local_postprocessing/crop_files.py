import click
import numpy as np
import nibabel as nb
from tqdm import tqdm
import os


@click.command()
@click.argument("files", nargs=-1)
def main(files):
    min_x = np.zeros(len(files), dtype=np.int)
    max_x = np.zeros_like(min_x)
    min_y = np.zeros_like(min_x)
    max_y = np.zeros_like(min_x)
    min_z = np.zeros_like(min_x)
    max_z = np.zeros_like(min_x)
    for i, file in enumerate(tqdm(files)):
        a = np.squeeze(nb.load(file).get_data())
        x, y, z = np.where(a != 0)
        min_x[i], max_x[i] = np.min(x), np.max(x)
        min_y[i], max_y[i] = np.min(y), np.max(y)
        min_z[i], max_z[i] = np.min(z), np.max(z)

    output_folder = "../data/cropped"
    try:
        os.makedirs(output_folder)
    except:
        pass
    print(np.min(min_x), np.max(max_x))
    print(np.min(min_y), np.max(max_y))
    print(np.min(min_z), np.max(max_z))
    for file in tqdm(files):
        basename = os.path.splitext(os.path.basename(file))[0]
        output_file = os.path.join(
            output_folder, basename) + "_cropped.nii.gz"
        output_mask = os.path.join(
            output_folder, basename) + "_cropped_mask.nii.gz"
        a = nb.load(file).get_data()[
            np.min(min_x):np.max(max_x),
            np.min(min_y):np.max(max_y),
            np.min(min_z):np.max(max_z)
        ]
        b = np.zeros_like(a, dtype=np.uint8)
        b[a > 0] = 1
        affine = np.eye(4)
        new_img = nb.Nifti1Image(a, affine)
        new_mask = nb.Nifti1Image(b, affine)
        nb.save(new_img, output_file)
        nb.save(new_mask, output_mask)


if __name__ == "__main__":
    main()
