from __future__ import division
import numpy as np
import scipy.ndimage
import scipy.weave
import logging


def solve_laplace(
        (file_name, data),
        csf_gm_threshold=650,
        gm_wm_threshold=1000,
        max_iter=1000,
        eps=1e-3):
    seed = np.zeros(data.shape, dtype=float)
    seed[np.logical_and(
        data > csf_gm_threshold,
        data < gm_wm_threshold)] = 500
    seed[data > 1000] = 1000
    mask = np.zeros_like(seed, dtype=np.bool)
    mask[seed == 500] = 1
    n_iter = 0
    e = 2 * eps
    for i in range(max_iter):
        nx, ny, nz = seed.shape
        code = """
        #line 1 "solve_laplace.py"
        double tmp;
        double err = 0;
        double diff;
        double sixth = 1. / 6;
        for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
        for (int k = 1; k < nz - 1; k++) {
            if (seed(i, j, k) == 0 or seed(i, j, k) == 1000) {
                continue;
            }
            tmp = seed(i, j, k);
            seed(i, j, k) = (
                seed(i - 1, j, k) + seed(i + 1, j, k) +
                seed(i, j - 1, k) + seed(i, j + 1, k) +
                seed(i, j, k - 1) + seed(i, j, k + 1)) * sixth;
            diff = seed(i, j, k) - tmp;
            err += diff * diff;
        }
        }
        }
        return_val = err;
        """
        e = scipy.weave.inline(
            code,
            ["seed", "nx", "ny", "nz"],
            type_converters=scipy.weave.converters.blitz,
            compiler="gcc"
        )
        n_iter += 1
        if e < eps: break
    return file_name, seed


def calculate_thickness(
        (file_name, potential),
        max_l):
    ex, ey, ez = np.gradient(potential)
    norm = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
    ex /= norm
    ey /= norm
    ez /= norm
    ex[~np.isfinite(ex)] = 0
    ey[~np.isfinite(ey)] = 0
    ez[~np.isfinite(ez)] = 0
    l = np.zeros_like(potential, dtype=int)
    nx, ny, nz = potential.shape
    code = """
    #line 1 "calculate_thickness.py"
    double phi;
    int l1;
    int l2;
    double x;
    double y;
    double z;
    int x_0;
    int y_0;
    int z_0;
    int x_1;
    int y_1;
    int z_1;
    for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
    for (int k = 1; k < nz - 1; k++) {
    phi = potential(i, j, k);
    if (phi == 0 or phi == 1000) continue;
    x = i;
    y = j;
    z = k;
    l1 = 0;
    while (phi > 0 and l1 < max_l) {
        x_0 = std::round(x);
        if (x_0 < 0 or x_0 > nx - 1) {
            l1 = max_l;
            break;
        }
        y_0 = std::round(y);
        if (y_0 < 0 or y_0 > ny - 1) {
            l1 = max_l;
            break;
        }
        z_0 = std::round(z);
        if (z_0 < 0 or z_0 > nz - 1) {
            l1 = max_l;
            break;
        }
        x -= ex(x_0, y_0, z_0);
        y -= ey(x_0, y_0, z_0);
        z -= ez(x_0, y_0, z_0);
        x_1 = std::round(x);
        if (x_1 < 0 or x_1 > nx - 1) {
            l1 = max_l;
            break;
        } y_1 = std::round(y);
        if (y_1 < 0 or y_1 > ny - 1) {
            l1 = max_l;
            break;
        }
        z_1 = std::round(z);
        if (z_1 < 0 or z_1 > nz - 1) {
            l1 = max_l;
            break;
        }
        phi = potential(x_1, y_1, z_1);
        l1 += 1;
    }
    if (l1 >= max_l) continue;
    phi = potential(i, j, k);
    x = i;
    y = j;
    z = k;
    l2 = 0;
    while (phi < 1000 and l2 < max_l) {
        x_0 = std::round(x);
        if (x_0 < 0 or x_0 > nx - 1) {
            l2 = max_l;
            break;
        }
        y_0 = std::round(y);
        if (y_0 < 0 or y_0 > ny - 1) {
            l2 = max_l;
            break;
        }
        z_0 = std::round(z);
        if (z_0 < 0 or z_0 > nz - 1) {
            l2 = max_l;
            break;
        }
        x += ex(x_0, y_0, z_0);
        y += ey(x_0, y_0, z_0);
        z += ez(x_0, y_0, z_0);
        x_1 = std::round(x);
        if (x_1 < 0 or x_1 > nx - 1) {
            l2 = max_l;
            break;
        }
        y_1 = std::round(y);
        if (y_1 < 0 or y_1 > ny - 1) {
            l2 = max_l;
            break;
        }
        z_1 = std::round(z);
        if (z_1 < 0 or z_1 > nz - 1) {
            l2 = max_l;
            break;
        }
        phi = potential(x_1, y_1, z_1);
        l2 += 1;
    }
    if (l2 >= max_l) continue;
    l(i, j, k) = l1 + l2;
    }
    }
    }
    """
    scipy.weave.inline(
        code,
        ["potential", "ex", "ey", "ez", "l", "nx", "ny", "nz", "max_l"],
        type_converters=scipy.weave.converters.blitz,
        compiler="gcc",
        extra_compile_args=['-std=c++11'],
    )
    return file_name, l
