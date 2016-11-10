from __future__ import division, print_function
import os
import zlib
import io
import numpy as np
import click
import tqdm
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import itk
import nibabel as nb

from healthybrains.inputoutput import id_from_file_name

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
@click.argument("warp_field", nargs=1)
@click.argument("warped_atlas", nargs=1)
def main(warp_field, warped_atlas):
    wbits = zlib.MAX_WBITS | 16
    field = nb.load(warp_field).get_data()
    atlas = nb.load(warped_atlas).get_data()
    distances = np.linalg.norm(field, axis=3)
    for region_name, desikan_label_id in desikan_label_ids.iteritems():
        print(distances.shape)
        print(field.shape)
        print(atlas.shape)
        region = atlas.astype(np.uint8)
        region[region != desikan_label_id] = 0
        region_vtk = numpy_to_vtk(
            np.ravel(region, order="F"),
            array_type=vtk.VTK_UNSIGNED_SHORT)
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(region.shape)
        # # vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
        vtk_image.GetPointData().SetScalars(region_vtk)

        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputData(vtk_image)
        dmc.GenerateValues(1, desikan_label_id, desikan_label_id)

        normals = vtk.vtkPolyDataNormals()
        normals.SetAutoOrientNormals(1)
        normals.SetInputConnection(dmc.GetOutputPort())
        normals.Update()

        polygon = normals.GetOutput()
        n = polygon.GetNumberOfPoints()
        distance = vtk.vtkFloatArray()
        distance.SetName("distance")
        for i in range(n):
            point = polygon.GetPoint(i)
            normal = polygon.GetPointData().GetNormals().GetTuple(i)
            x, y, z = int(point[0]), int(point[1]), int(point[2])
            d = distances[x, y, z]
            sign = np.dot(field[x, y, z], np.array(normal))
            sign = 1 if sign >= 0 else -1
            distance.InsertNextValue(d * sign)
        polygon.GetPointData().AddArray(distance)
        polygon.GetPointData().SetActiveScalars("distance")

        mesh_file_name = warp_field.replace(
            "warp_field.nii.gz", region_name + "_mesh.vtk")
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polygon)
        writer.SetFileName(mesh_file_name)
        writer.Update()

        itk_reader = itk.MeshFileReader()

        min_d, max_d = distance.GetRange()
        colorLookupTable= vtk.vtkLookupTable()
        colorLookupTable.SetHueRange(2 / 3, 1)
        #colorLookupTable.SetSaturationRange(0, 0)
        #colorLookupTable.SetValueRange(1, 0)
        #colorLookupTable.SetNumberOfColors(256) #256 default
        colorLookupTable.Build()

        # render
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polygon)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarRange(min_d, max_d)
        mapper.SetScalarModeToUsePointData()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        renderer_interaction = vtk.vtkRenderWindowInteractor()
        renderer_interaction.SetRenderWindow(render_window)
        renderer.AddActor(actor)
        renderer.SetBackground(0, 0, 0)
        render_window.SetSize(600, 600)
        def exit_check(obj, event):
            if obj.GetEventPending() != 0:
                obj.SetAbortRender(1)
        
        # Tell the application to use the function as an exit check.
        render_window.AddObserver("AbortCheckEvent", exit_check)
        
        renderer_interaction.Initialize()
        # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
        render_window.Render()
        renderer_interaction.Start()
        break
    # np.save(output, arrays)


if __name__ == "__main__":
    main()
