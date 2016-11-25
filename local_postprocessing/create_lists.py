from __future__ import division, print_function
import os
import click

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
@click.argument("health", nargs=1, type=click.Path(exists=True))
def main(health):
    for region_name, label_id in desikan_label_ids.iteritems():
        folder = "../data/lists/{0}".format(region_name)
        try:
            os.makedirs(folder)
        except:
            pass
        output0_name = folder + "/list0.txt"
        output1_name = output0_name.replace("0", "1")
        output0_file = open(output0_name, "w")
        output1_file = open(output1_name, "w")
        with open(health, "r") as health_file:
            for i, line in enumerate(health_file):
                health_status = int(line)
                output_line = "/tenochtitlan/data/fsl/train_{0}_cropped/train_{0}_cropped_{1}.nrrd".format(
                    i + 1, region_name)
                if health_status == 0:
                    output0_file.write(output_line)
                    output0_file.write("\n")
                else:
                    output1_file.write(output_line)
                    output1_file.write("\n")
        

if __name__ == "__main__":
    main()
