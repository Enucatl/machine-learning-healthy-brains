import io
import nibabel as nb
import numpy as np
import apache_beam as beam


class _Nifti1Source(beam.io.filebasedsource.FileBasedSource):

    def __init__(self, file_pattern, min_bundle_size):
        super(_Nifti1Source, self).__init__(
            file_pattern=file_pattern,
            min_bundle_size=min_bundle_size,
            splittable=False)

    def read_records(self, file_name, range_tracker):
        with self.open_file(file_name) as f:
            hdr_fh = nb.fileholders.FileHolder(fileobj=f)
            header = nb.Nifti1Image.header_class.from_fileobj(f)
            array_proxy = header.data_from_fileobj(f)
            data = array_proxy[..., 0]
            yield (file_name, data)


class ReadNifti1(beam.transforms.PTransform):

    def __init__(self,
                 file_pattern=None,
                 min_bundle_size=0
                ):
        super(ReadNifti1, self).__init__()
        self._file_pattern = file_pattern
        self._min_bundle_size = min_bundle_size

    def apply(self, pcoll):
        return pcoll.pipeline | beam.io.Read(
            _Nifti1Source(
                file_pattern=self._file_pattern,
                min_bundle_size=self._min_bundle_size))


def fake_data((file_name, data)):
    return file_name, np.arange(10), np.arange(10)


def thickness_data_to_string((file_name, thickness)):
    output_string = io.BytesIO()
    np.save(output_string, np.array([file_name]))
    np.save(output_string, thickness)
    final_string = output_string.getvalue()
    return final_string
