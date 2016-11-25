import numpy as np
import click
import tqdm
import nibabel as nb
import nrrd

desikan_label_ids = {
    "left_thalamus": 10,
    "right_thalamus": 49,
    "left_caudate": 11,
    "right_caudate": 50,
    "left_putamen": 12,
    "right_putamen": 51,
    "left_pallidum": 13,
    "right_pallidum": 52,
    "left_hippocampus": 17,
    "right_hippocampus": 53,
    "left_amygdala": 18,
    "right_amygdala": 54,
}


@click.command()
@click.argument("warped_atlas", nargs=1)
def main(warped_atlas):
    atlas = nb.load(warped_atlas).get_data()
    margin = 5
    for region_name, label_id in desikan_label_ids.iteritems():
        output_name = warped_atlas.replace(
            "_warped_atlas.nii.gz", "_{0}.nrrd".format(region_name))
        x, y, z = np.where(atlas == label_id)
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        min_z, max_z = np.min(z), np.max(z)
        output_data = np.zeros(
            (2 * margin + max_x - min_x,
             2 * margin + max_y - min_y,
             2 * margin + max_z - min_z),
            dtype=np.int16)
        region_atlas = atlas.copy()
        region_atlas[region_atlas != label_id] = 0
        output_data[...] = region_atlas[
            min_x - margin:max_x + margin,
            min_y - margin:max_y + margin,
            min_z - margin:max_z + margin]
        nrrd.write(output_name, output_data)
    


if __name__ == "__main__":
    main()
