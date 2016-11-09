from __future__ import division, print_function
import os
import click
import nibabel as nb
import numpy as np
from subprocess import check_call


desikan_label_ids = {
    # "left_thalamus": 10,
    # "right_thalamus": 49,
    # "left_caudate": 11,
    # "right_caudate": 50,
    # "left_putamen": 12,
    # "right_putamen": 51,
    # "left_pallidum": 13,
    # "right_pallidum": 52,
    "left_hippocampus": 17,
    # "right_hippocampus": 53,
    # "left_amygdala": 18,
    # "right_amygdala": 54,
}


@click.command()
@click.argument("input_file", nargs=1)
def main(input_file):
    basename = os.path.splitext(os.path.basename(input_file))[0]
    print(basename)
    output_dir = "../data/fsl/{0}".format(basename)
    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    omat = os.path.join(output_dir, basename) + ".mat"
    out = os.path.join(output_dir, basename) + "_flirt.nii.gz"
    # ref_name = "../data/IIT_GM_Desikan_atlas_cropped.nii.gz"
    ref_name = "../data/IITmean_t1_cropped.nii.gz"
    flirt_command = "/usr/share/fsl/5.0/bin/flirt\
    -in {input_file} -ref {ref}\
    -out {out} -omat {omat}\
    -bins 256 -cost mutualinfo\
    -searchrx 0 0 -searchry 0 0 -searchrz 0 0\
    -dof 12  -interp trilinear".format(
        input_file=input_file,
        out=out,
        omat=omat,
        ref=ref_name,
        )
    # check_call(flirt_command, shell=True)
    atlas_name = "../data/IIT_GM_Desikan_atlas_cropped.nii.gz"
    affine = np.genfromtxt(omat).astype(int)
    for region_name, desikan_label_id in desikan_label_ids.iteritems():
        atlas = nb.load(atlas_name).get_data()
        ref = nb.load(ref_name).get_data()
        atlas[atlas != desikan_label_id] = 0
        x, y, z = np.where(atlas == desikan_label_id)
        margin = 15
        x_min, x_max = np.min(x) - margin, np.max(x) + margin
        y_min, y_max = np.min(y) - margin, np.max(y) + margin
        z_min, z_max = np.min(z) - margin, np.max(z) + margin
        x_0 = affine[0, 3]
        y_0 = affine[1, 3]
        z_0 = affine[2, 3]
        ref_image = ref[
            x_min:x_max,
            y_min:y_max,
            z_min:z_max
        ]
        image = nb.load(input_file).get_data()[
            x_min - x_0:x_max - x_0,
            y_min - y_0:y_max - y_0,
            z_min - z_0:z_max - z_0,
        ]
        region_ref_name = os.path.join(
            output_dir, region_name + "_atlas.nii.gz")
        region_image_name = os.path.join(
            output_dir, region_name + "_" + basename + ".nii.gz")
        flirt_region_image_name = os.path.join(
            output_dir, region_name + "_" + basename + "_flirt.nii.gz")
        nb.save(nb.Nifti1Image(ref_image, np.eye(4)), region_ref_name)
        nb.save(nb.Nifti1Image(image, np.eye(4)), region_image_name)
        region_mat_name = region_image_name.replace(".nii.gz", ".mat")
        flirt_command = "/usr/share/fsl/5.0/bin/flirt\
        -in {input_file} -ref {ref}\
        -out {out} -omat {omat}\
        -bins 256 -cost mutualinfo\
        -searchrx -20 20 -searchry -20 20 -searchrz -20 20\
        -dof 12  -interp trilinear".format(
            input_file=region_image_name,
            out=flirt_region_image_name,
            omat=region_mat_name,
            ref=region_ref_name,
            )
        check_call(flirt_command, shell=True)


if __name__ == "__main__":
    main()
