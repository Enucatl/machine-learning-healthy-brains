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
    -in {ref} -ref {input_file}\
    -out {out} -omat {omat}\
    -bins 256 -cost mutualinfo\
    -searchrx 0 0 -searchry 0 0 -searchrz 0 0\
    -dof 12  -interp trilinear".format(
        input_file=input_file,
        out=out,
        omat=omat,
        ref=ref_name,
        )
    print(flirt_command)
    # check_call(flirt_command, shell=True)
    print("done.")
    atlas_name = "../data/IIT_GM_Desikan_atlas_cropped.nii.gz"
    affine = np.genfromtxt(omat).astype(int)
    atlas = nb.load(atlas_name).get_data()
    ref = nb.load(ref_name).get_data()
    fnirt_image_name = os.path.join(
        output_dir, basename + "_fnirt.nii.gz")
    coefficients_file_name = os.path.join(
        output_dir, basename + "_coefficients.nii.gz")
    warp_field_name = os.path.join(
        output_dir, basename + "_warp_field.nii.gz")
    fnirt_command = "/usr/share/fsl/5.0/bin/fnirt\
    --in={ref} --ref={input_file}\
    --iout={out} --aff={aff}\
    --cout={coefficients_file_name}\
    --fout={fout}\
    --warpres=10,10,10\
    --subsamp=8,4,2,1\
    --reffwhm=2,0,0,0\
    --lambda=300,100,50,25\
    --intmod=global_linear\
    ".format(
        input_file=input_file,
        ref=ref_name,
        out=fnirt_image_name,
        aff=omat,
        coefficients_file_name=coefficients_file_name,
        fout=warp_field_name,
        )
    print(fnirt_command)
    # check_call(fnirt_command, shell=True)
    warped_atlas = os.path.join(
        output_dir, basename + "_warped_atlas.nii.gz")
    applyward_command = "/usr/lib/fsl/5.0/applywarp\
    --ref={ref} --in={atlas_name} --warp={warp}\
    --out={warped_atlas} --interp=nn\
    ".format(
        ref=input_file,
        atlas_name=atlas_name,
        warp=coefficients_file_name,
        warped_atlas=warped_atlas
    )
    print(applyward_command)
    check_call(applyward_command, shell=True)
    print("done.")


if __name__ == "__main__":
    main()
