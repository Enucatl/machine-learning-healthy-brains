import apache_beam as beam
import healthybrains
import healthybrains.inputoutput
import healthybrains.thickness


class ThicknessOptions(beam.utils.options.PipelineOptions):

    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            "--input",
            dest="input",
            default="data/set_train/train_10[01].nii"
        )
        parser.add_argument(
            "--output",
            dest="output",
            default="output/cortical_thickness"
        )
        parser.add_argument(
            "--zoom",
            dest="zoom",
            type=int,
            default=1
        )


if __name__ == "__main__":
    pipeline_options = beam.utils.options.PipelineOptions()
    p = beam.Pipeline(options=pipeline_options)
    options = pipeline_options.view_as(ThicknessOptions)
    max_l = 20
    datasets = (
        p
        | "ReadDataset" >> healthybrains.inputoutput.ReadNifti1(options.input)
        | beam.core.Map(healthybrains.thickness.get_frontal_region)
        | beam.core.Map(healthybrains.thickness.zoom, options.zoom)
        | beam.core.Map(healthybrains.thickness.solve_laplace)
        | beam.core.Map(
            healthybrains.thickness.calculate_thickness, 20 * options.zoom)
        | beam.core.Map(healthybrains.inputoutput.thickness_data_to_string)
        | beam.io.WriteToText(
            options.output,
            file_name_suffix=".npy.gz",
            coder=beam.coders.BytesCoder(),
            append_trailing_newlines=False,
            compression_type=beam.io.fileio.CompressionTypes.GZIP,
        )
    )
    p.run()
