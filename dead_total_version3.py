import numpy as np
from skimage.morphology import skeletonize

from search_dead_cell_version3 import search_dead_cell
import pyvista as pv
from connectionTest import correct_flood
import sys


def reshape(a, b, c):
    a = np.array(a).T
    b = np.array(b).T
    c = np.array(c).T
    m = np.concatenate((a, b, c), axis=1)
    return m


def dead_total(LiverPath, ArteryPath, VeinPath, TumourPath, i, x_l1, y_l1, z_l1, x_l2, y_l2, z_l2,res=[]):
    liver1 = np.load(LiverPath)
    liver2 = np.load(ArteryPath)
    liver3 = np.load(VeinPath)
    liver5 = np.load(TumourPath)

    liver2[liver2 == 2] = 1
    liver3[liver3 == 3] = 1
    liver3 = skeletonize(liver3)
    liver3[liver3 == 255] = 3
    liver2 = skeletonize(liver2)
    liver2[liver2 == 255] = 2

    matrix = np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))
    matrix += liver1
    matrix += liver2
    matrix[matrix == 3] = 2
    matrix += liver3
    matrix[matrix == 5] = 3
    matrix[matrix == 4] = 3
    print(matrix[x_l1][y_l1][z_l1])
    print(matrix[x_l2][y_l2][z_l2])

    if matrix[x_l1][y_l1][z_l1] != 2:
        print('门静脉点不正确')
        sys.exit()
    if matrix[x_l2][y_l2][z_l2] != 3:
        print('肝静脉点不正确')
        sys.exit()

    correct_flood(matrix, 2, x_l1, y_l1, z_l1)
    correct_flood(matrix, 3, x_l2, y_l2, z_l2)

    for p in liver5:
        if matrix[p[0], p[1], p[2]] == 2:
            matrix[p[0], p[1], p[2]] = 7
        elif matrix[p[0], p[1], p[2]] == 3:
            matrix[p[0], p[1], p[2]] = 8
        else:
            matrix[p[0], p[1], p[2]] = 5

    for p in res:
        if matrix[p[0], p[1], p[2]] == 2:
            matrix[p[0], p[1], p[2]] = 7
        elif matrix[p[0], p[1], p[2]] == 3:
            matrix[p[0], p[1], p[2]] = 8
        else:
            matrix[p[0], p[1], p[2]] = 5

    l_5 = reshape([np.where(matrix == 5)[0]], [np.where(matrix == 5)[1]], [np.where(matrix == 5)[2]])

    index = np.load('data/vessel_to_point.npy',allow_pickle=True).item()
    print('索引对应表加载完成')
    a1 = search_dead_cell(matrix, index, 2, x_l1, y_l1, z_l1, i,l_5)
    if(len(res)==0):
        np.save('data/缺血区域' + str(i) + '.npy', a1)
    print("缺血体积:", len(a1) * 1.2 * 1.2 * 1.2, 'mm^3')
    a2 = search_dead_cell(matrix, index, 3, x_l2, y_l2, z_l2, i,l_5)
    if(len(res)==0):
        np.save('data/淤血区域' + str(i) + '.npy', a2)
    print("淤血体积:", len(a2) * 1.2 * 1.2 * 1.2, 'mm^3')

    if not a1.any() and not a2.any():
        total_dead = []
    elif not a2.any():
        total_dead = a1
    elif not a1.any():
        total_dead = a2
    else:
        a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
        a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
        total_dead = np.union1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])

    np.save('data/total_dead' + str(i) + '.npy', total_dead)



