import nibabel as nb
import numpy as np
import click


@click.command()
@click.argument("files", nargs=-1)
def main(files):
    for file_name in files:
        affine = np.eye(4)
        img = nb.load(file_name).get_data()[
            3:-3, :-10, :-6]
        new_img = nb.Nifti1Image(img, affine)
        output_file_name = file_name.replace(
            ".nii.gz", "_cropped.nii.gz")
        nb.save(new_img, output_file_name)


if __name__ == "__main__":
    main()
