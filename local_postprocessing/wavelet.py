from __future__ import division, print_function
import os
from glob import glob
import zlib
import io
import numpy as np
import click
import tqdm
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import itk

from healthybrains.inputoutput import id_from_file_name


@click.command()
@click.argument("folder", nargs=1)
@click.argument("output", nargs=1)
@click.option("--zoom", type=int, default=1)
def main(folder, output, zoom):
    max_l = 20 * zoom
    file_names = glob(os.path.join(folder, "*"))
    wbits = zlib.MAX_WBITS | 16
    arrays = np.zeros(shape=(len(file_names), max_l - 2))
    for file_name in tqdm.tqdm(file_names):
        decompressor = zlib.decompressobj(wbits)
        with open(file_name, "rb") as infile:
            stringio = io.BytesIO()
            decompressed = decompressor.decompress(infile.read())
            stringio.write(decompressed)
            stringio.seek(0)
            name = id_from_file_name(np.load(stringio)[0])
            thicknesses = np.load(stringio)
            thicknesses_vtk = numpy_to_vtk(
                np.ravel(thicknesses, order="F"),
                array_type=vtk.VTK_UNSIGNED_SHORT)
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(thicknesses.shape)
            # vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
            vtk_image.GetPointData().SetScalars(thicknesses_vtk)

            #threshold
            threshold = vtk.vtkImageThreshold()
            threshold.SetInputData(vtk_image)
            threshold.ThresholdBetween(1, 20)
            threshold.ReplaceInOn()
            threshold.SetInValue(20)
            threshold.ReplaceOutOn()
            threshold.SetOutValue(0)
            threshold.Update()

            # alpha_channel_function = vtk.vtkPiecewiseFunction()
            # alpha_channel_function.AddPoint(0, 0)
            # alpha_channel_function.AddPoint(100, 1)
            # color_function = vtk.vtkColorTransferFunction()
            # color_function.AddRGBPoint(50, 1, 0, 0)
            # volume_property = vtk.vtkVolumeProperty()
            # volume_property.SetColor(color_function)
            # volume_property.SetScalarOpacity(alpha_channel_function)
            # volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
            # volume_mapper.SetInputConnection(threshold.GetOutputPort())
            # # volume_mapper.SetInputData(vtk_image)
            # volume = vtk.vtkVolume()
            # volume.SetMapper(volume_mapper)
            # volume.SetProperty(volume_property)
            # renderer = vtk.vtkRenderer()
            # render_window = vtk.vtkRenderWindow()
            # render_window.AddRenderer(renderer)
            # renderer_interaction = vtk.vtkRenderWindowInteractor()
            # renderer_interaction.SetRenderWindow(render_window)
            # renderer.AddVolume(volume)
            # renderer.SetBackground(0, 0, 0)
            # render_window.SetSize(1200, 1200)
            # def exit_check(obj, event):
                # if obj.GetEventPending() != 0:
                    # obj.SetAbortRender(1)

            dmc = vtk.vtkDiscreteMarchingCubes()
            dmc.SetInputConnection(threshold.GetOutputPort())
            dmc.GenerateValues(1, 20, 20)
            dmc.Update()

            # render
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(dmc.GetOutputPort())
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
