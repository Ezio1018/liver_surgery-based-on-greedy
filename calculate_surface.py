import numpy as np
import pyvista as pv


def calculate_surface(f, value):
    surface2 = f
    matrix_surface = np.argwhere(f == value)
    matrix = np.zeros((surface2.shape[0], surface2.shape[1], surface2.shape[2]))
    x = np.zeros((1, surface2.shape[1], surface2.shape[2]))
    list_matrix1 = np.vstack((x, surface2))
    list_matrix1 = list_matrix1[:-1, :, :]
    list_matrix2 = np.vstack((surface2, x))
    list_matrix2 = list_matrix2[1:, :, :]
    y = np.zeros((surface2.shape[0], 1, surface2.shape[2]))
    list_matrix3 = np.hstack((y, surface2))
    list_matrix3 = list_matrix3[:, :-1, :]
    list_matrix4 = np.hstack((surface2, y))
    list_matrix4 = list_matrix4[:, 1:, :]
    z = np.zeros((surface2.shape[0], surface2.shape[1], 1))
    list_matrix5 = np.c_[z, surface2]
    list_matrix5 = list_matrix5[:, :, :-1]
    list_matrix6 = np.c_[surface2, z]
    list_matrix6 = list_matrix6[:, :, 1:]

    matrix = matrix + list_matrix1 + list_matrix2 + list_matrix3 + list_matrix4 + list_matrix5 + list_matrix6
    matrix1 = np.argwhere(matrix == 6*value)
    matrix_surface = matrix_surface.tolist()
    matrix1 = matrix1.tolist()
    a1 = np.asarray(matrix_surface)
    a2 = np.asarray(matrix1)
    if not a1.any() and not a2.any():
        new_list = []
    elif not a2.any():
        new_list = a1
    elif not a1.any():
        new_list = a2
    else:
        a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
        a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
        new_list = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])

    return new_list
