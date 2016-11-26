import os
import click
import numpy as np
import vtk
from tqdm import tqdm
import pandas as pd


@click.command()
@click.argument("file_name", nargs=1)
def main(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    polygon = reader.GetOutput()
    n = polygon.GetNumberOfPoints()
    frontal = []
    central = []
    cerebellum = []
    thickness = polygon.GetPointData().GetScalars()
    for i in range(n):
        x, y, z = polygon.GetPoint(i)
        if (x > 40 and x < 130 and
            y > 140 and y < 190 and
            z > 45 and z < 160):
            t = thickness.GetValue(i)
            frontal.append(t)
        elif (x > 25 and x < 150 and
            y > 70 and y < 125 and
            z > 75 and z < 150):
            t = thickness.GetValue(i)
            central.append(t)
        elif (x > 40 and x < 130 and
            y > 36 and y < 90 and
            z > 15 and z < 40):
            t = thickness.GetValue(i)
            cerebellum.append(t)
        else: continue
    frontal = np.array(frontal, dtype=np.uint8)
    central = np.array(central, dtype=np.uint8)
    cerebellum = np.array(cerebellum, dtype=np.uint8)
    frontal = frontal[frontal > 0]
    central = central[central > 0]
    cerebellum = cerebellum[cerebellum > 0]
    max_t = 12
    bins = np.arange(1, max_t + 1)
    frontal_hist, _ = np.histogram(frontal, bins, density=True)
    central_hist, _ = np.histogram(central, bins, density=True)
    cerebellum_hist, _ = np.histogram(cerebellum, bins, density=True)
    frontal_central_hist, _ = np.histogram(
        np.concatenate((frontal, central)), bins, density=True)
    frontal_cumulative = np.cumsum(frontal_hist)
    central_cumulative = np.cumsum(central_hist)
    frontal_central_cumulative = np.cumsum(frontal_central_hist)
    cerebellum_cumulative = np.cumsum(cerebellum_hist)
    folder_name = os.path.dirname(file_name).replace(
        "frontal_thickness", "thickness_histograms")
    output_name = os.path.splitext(os.path.basename(file_name))[0]
    output_name = os.path.join(folder_name, output_name)
    np.save(output_name + "_frontal_hist.npy", frontal_hist)
    np.save(output_name + "_central_hist.npy", central_hist)
    np.save(output_name + "_cerebellum_hist.npy", cerebellum_hist)
    np.save(output_name + "_frontal_cumulative.npy", frontal_cumulative)
    np.save(output_name + "_central_cumulative.npy", central_cumulative)
    np.save(output_name + "_cerebellum_cumulative.npy", cerebellum_cumulative)
    np.save(
        output_name + "_frontal_central_cumulative.npy",
        frontal_central_cumulative)


if __name__ == "__main__":
    main()
