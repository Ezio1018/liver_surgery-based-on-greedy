from plyfile import PlyData, PlyElement
import numpy as np
from calculate_surface import calculate_surface


def toply(live_path,blood_1_path,blood_2_path,cut_cell_path,tumour_path,i):

    def write_ply(save_path, points, text=True):
        """
        save_path : path to save: '/yy/XX.ply'
        pt: point_cloud: size (N,3)
        """
        points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=text).write(save_path)


    liver1 = np.load('data/liver.npy')
    matrix = np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))
    liver = np.load(live_path)
    for p in liver:
        matrix[p[0], p[1], p[2]] = 1
    l1 = calculate_surface(matrix, 1)
    write_ply("stl/liver_remain"+str(i)+".ply", l1)
    print('肝实质保存成功')
    blood1 = np.load(blood_1_path)
    for p in blood1:
        matrix[p[0], p[1], p[2]] = 2
    l2 = calculate_surface(matrix, 2)
    write_ply("stl/m_vessel"+str(i)+".ply", l2)
    print('门静脉保存成功')
    blood2 = np.load(blood_2_path)
    for p in blood2:
        matrix[p[0], p[1], p[2]] = 3
    l3 = calculate_surface(matrix, 3)
    write_ply("stl/g_vessel"+str(i)+".ply", l3)
    print('肝静脉保存成功')
    cut_cell = np.load(cut_cell_path)
    # print(cut_cell)
    for p in cut_cell:
        matrix[p[0], p[1], p[2]] = 4
    l4 = calculate_surface(matrix, 4)
    write_ply("stl/liver_cut"+str(i)+".ply", l4)
    print('切除细胞保存成功')

    tumour_cell = np.load(tumour_path)
    l5 = calculate_surface(tumour_cell, 4)
    # print(len(l5))
    write_ply("stl/tumor"+str(i)+".ply", l5)
    print('随机占位保存成功')
