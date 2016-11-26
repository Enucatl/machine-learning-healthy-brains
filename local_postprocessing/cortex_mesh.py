from __future__ import division, print_function
import os
import zlib
import io
import numpy as np
from tqdm import tqdm
import click
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import itk
import nibabel as nb

from healthybrains.inputoutput import id_from_file_name

@click.command()
@click.argument("file_names", nargs=-1)
@click.option("--targets", nargs=1, default="../data/targets.csv")
@click.option("--show", is_flag=True)
def main(file_names, targets, show):
    for file_name in tqdm(file_names):
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
        health = int(open(targets, "r").readlines()[file_id - 1])
        cortex = np.load(stringio)
        cortex[cortex > 12] = 0
        cortex_mask = np.zeros_like(cortex, dtype=np.uint8)
        cortex_mask[cortex > 0] = 1
        region_vtk = numpy_to_vtk(
            np.ravel(cortex_mask, order="F"),
            array_type=vtk.VTK_UNSIGNED_SHORT)
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(cortex.shape)
        vtk_image.GetPointData().SetScalars(region_vtk)

        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputData(vtk_image)
        dmc.GenerateValues(1, 1, 1)

        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputConnection(dmc.GetOutputPort())
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.Update()

        polygon = connectivity_filter.GetOutput()
        n = polygon.GetNumberOfPoints()
        thickness = vtk.vtkFloatArray()
        thickness.SetName("thickness")
        d = 0
        for i in range(n):
            point = polygon.GetPoint(i)
            x, y, z = int(point[0]), int(point[1]), int(point[2])
            if cortex[x, y, z] > 0:
                d = cortex[x, y, z]
            thickness.InsertNextValue(d)
        polygon.GetPointData().AddArray(thickness)
        polygon.GetPointData().SetActiveScalars("thickness")

        mesh_file_name = file_name.replace(
            ".npy.gz", "_mesh_{0}_{1}.vtk".format(
            file_id, health))
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polygon)
        writer.SetFileName(mesh_file_name)
        writer.Update()

        if show:
            min_d, max_d = 0, 6
            print(min_d, max_d)
            num_colors = 256
            color_table = vtk.vtkLookupTable()
            color_table.SetNumberOfTableValues(num_colors)
            transfer_func = vtk.vtkColorTransferFunction()
            transfer_func.SetColorSpaceToDiverging()
            transfer_func.AddRGBPoint(0, 0.230, 0.299,  0.754)
            transfer_func.AddRGBPoint(1, 0.706, 0.016, 0.150)
            for ii, ss in enumerate([float(xx)/float(num_colors) for xx in range(num_colors)]):
                cc = transfer_func.GetColor(ss)
                color_table.SetTableValue(ii, cc[0], cc[1], cc[2], 1.0)
            color_table.SetTableRange(min_d, max_d)
            

            # render
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polygon)
            mapper.SetScalarRange(min_d, max_d)
            mapper.ScalarVisibilityOn()
            mapper.SetLookupTable(color_table)
            mapper.SetScalarModeToUsePointData()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            color_bar = vtk.vtkScalarBarActor()
            color_bar.SetLookupTable(mapper.GetLookupTable())
            color_bar.SetTitle("thickness {0} {1}".format(file_id, health))
            color_bar.SetNumberOfLabels(6)
            renderer = vtk.vtkRenderer()
            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            renderer_interaction = vtk.vtkRenderWindowInteractor()
            renderer_interaction.SetRenderWindow(render_window)
            renderer.AddActor(actor)
            renderer.AddActor2D(color_bar)
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
            # np.save(output, arrays)


if __name__ == "__main__":
    main()
