import pyvista as pv
import numpy as np
import sys


def voxelize(LiverPath, ArteryPath, VeinPath, TumourPath, EXTumourPath):
    surface1 = pv.read(LiverPath)
    surface2 = pv.read(ArteryPath)
    surface3 = pv.read(VeinPath)
    surface4 = pv.read(TumourPath)
    surface5 = pv.read(EXTumourPath)
    surface6 = pv.read("weijie_file/liver_cut.stl")
    step = 1.2
    voxels1 = pv.voxelize(surface1, density=step, check_surface=False)
    print('肝实质体素化完成')
    voxels2 = pv.voxelize(surface2, density=step, check_surface=False)
    print('门静脉体素化完成')
    voxels3 = pv.voxelize(surface3, density=step, check_surface=False)
    print('肝静脉体素化完成')
    voxels4 = pv.voxelize(surface4, density=step, check_surface=False)
    print('占位体素化完成')
    voxels5 = pv.voxelize(surface5, density=step, check_surface=False)
    print('占位扩充体素化完成')
    voxels6 = pv.voxelize(surface6, density=step, check_surface=False)
    print('weipu_file体素化完成')
    x_min1, x_max1, y_min1, y_max1, z_min1, z_max1 = voxels1.bounds
    x_min2, x_max2, y_min2, y_max2, z_min2, z_max2 = voxels2.bounds
    x_min3, x_max3, y_min3, y_max3, z_min3, z_max3 = voxels3.bounds
    x_min4, x_max4, y_min4, y_max4, z_min4, z_max4 = voxels4.bounds
    x_min5, x_max5, y_min5, y_max5, z_min5, z_max5 = voxels5.bounds
    x_min6, x_max6, y_min6, y_max6, z_min6, z_max6 = voxels6.bounds

    x_min = min([x_min1, x_min2, x_min3])
    y_min = min([y_min1, y_min2, y_min3])
    z_min = min([z_min1, z_min2, z_min3])
    x_max = max([x_max1, x_max2, x_max3])
    y_max = max([y_max1, y_max2, y_max3])
    z_max = max([z_max1, z_max2, z_max3])
    print(x_min)
    print(y_min)
    print(z_min)

    a = np.array(voxels1.points)
    b = np.array(voxels2.points)
    c = np.array(voxels3.points)
    d = np.array(voxels4.points)
    e = np.array(voxels5.points)
    f = np.array(voxels6.points)
    x_group = np.arange(x_min, x_max, step)
    x_group = np.append(x_group, x_max)
    y_group = np.arange(y_min, y_max, step)
    y_group = np.append(y_group, y_max)
    z_group = np.arange(z_min, z_max, step)
    z_group = np.append(z_group, z_max)

    x_group1 = np.arange(x_min1, x_max1, step)
    x_group1 = np.append(x_group1, x_max1)
    y_group1 = np.arange(y_min1, y_max1, step)
    y_group1 = np.append(y_group1, y_max1)
    z_group1 = np.arange(z_min1, z_max1, step)
    z_group1 = np.append(z_group1, z_max1)

    x_group2 = np.arange(x_min2, x_max2, step)
    x_group2 = np.append(x_group2, x_max2)
    y_group2 = np.arange(y_min2, y_max2, step)
    y_group2 = np.append(y_group2, y_max2)
    z_group2 = np.arange(z_min2, z_max2, step)
    z_group2 = np.append(z_group2, z_max2)

    x_group3 = np.arange(x_min3, x_max3, step)
    x_group3 = np.append(x_group3, x_max3)
    y_group3 = np.arange(y_min3, y_max3, step)
    y_group3 = np.append(y_group3, y_max3)
    z_group3 = np.arange(z_min3, z_max3, step)
    z_group3 = np.append(z_group3, z_max3)

    x_group4 = np.arange(x_min4, x_max4, step)
    x_group4 = np.append(x_group4, x_max4)
    y_group4 = np.arange(y_min4, y_max4, step)
    y_group4 = np.append(y_group4, y_max4)
    z_group4 = np.arange(z_min4, z_max4, step)
    z_group4 = np.append(z_group4, z_max4)

    x_group5 = np.arange(x_min5, x_max5, step)
    x_group5 = np.append(x_group5, x_max5)
    y_group5 = np.arange(y_min5, y_max5, step)
    y_group5 = np.append(y_group5, y_max5)
    z_group5 = np.arange(z_min5, z_max5, step)
    z_group5 = np.append(z_group5, z_max5)

    x_group6 = np.arange(x_min6, x_max6, step)
    x_group6 = np.append(x_group6, x_max6)
    y_group6 = np.arange(y_min6, y_max6, step)
    y_group6 = np.append(y_group6, y_max6)
    z_group6 = np.arange(z_min6, z_max6, step)
    z_group6 = np.append(z_group6, z_max6)
    # print(a)
    # print(x_group)

    matrix = np.zeros((len(x_group) + 2, len(y_group) + 2, len(z_group) + 2))
    print(matrix.shape)
    bias_x = round((x_min1 - x_min) / step)
    bias_y = round((y_min1 - y_min) / step)
    bias_z = round((z_min1 - z_min) / step)
    for p in a:
        x = np.where(x_group1 == p[0])[0][0] + bias_x
        y = np.where(y_group1 == p[1])[0][0] + bias_y
        z = np.where(z_group1 == p[2])[0][0] + bias_z
        matrix[x, y, z] = 1
    np.save('data/liver.npy', matrix)

    print(1)

    matrix = np.zeros((len(x_group) + 2, len(y_group) + 2, len(z_group) + 2))
    bias_x = round((x_min2 - x_min) / step)
    bias_y = round((y_min2 - y_min) / step)
    bias_z = round((z_min2 - z_min) / step)
    for p in b:
        x = np.where(x_group2 == p[0])[0][0] + bias_x
        y = np.where(y_group2 == p[1])[0][0] + bias_y
        z = np.where(z_group2 == p[2])[0][0] + bias_z
        matrix[x, y, z] = 2
    np.save('data/m_vessel.npy', matrix)
    print(2)

    matrix = np.zeros((len(x_group) + 2, len(y_group) + 2, len(z_group) + 2))
    bias_x = round((x_min3 - x_min) / step)
    bias_y = round((y_min3 - y_min) / step)
    bias_z = round((z_min3 - z_min) / step)
    for p in c:
        x = np.where(x_group3 == p[0])[0][0] + bias_x
        y = np.where(y_group3 == p[1])[0][0] + bias_y
        z = np.where(z_group3 == p[2])[0][0] + bias_z
        matrix[x, y, z] = 3
    np.save('data/g_vessel.npy', matrix)
    print(3)

    matrix = np.zeros((len(x_group) + 2, len(y_group) + 2, len(z_group) + 2))
    bias_x = round((x_min4 - x_min) / step)
    bias_y = round((y_min4 - y_min) / step)
    bias_z = round((z_min4 - z_min) / step)
    for p in d:
        x = np.where(x_group4 == p[0])[0][0] + bias_x
        y = np.where(y_group4 == p[1])[0][0] + bias_y
        z = np.where(z_group4 == p[2])[0][0] + bias_z
        matrix[x, y, z] = 4
    np.save('data/tumor.npy', matrix)
    print(4)
    matrix = np.zeros((len(x_group) + 2, len(y_group) + 2, len(z_group) + 2))
    bias_x = round((x_min5 - x_min) / step)
    bias_y = round((y_min5 - y_min) / step)
    bias_z = round((z_min5 - z_min) / step)
    for p in e:
        x = np.where(x_group5 == p[0])[0][0] + bias_x
        y = np.where(y_group5 == p[1])[0][0] + bias_y
        z = np.where(z_group5 == p[2])[0][0] + bias_z
        # print((x,y,z))
        if(x>=matrix.shape[0] or y>=matrix.shape[1] or z>=matrix.shape[2] or x<0 or y<0 or z<0):
            # print("over")
            continue
        matrix[x, y, z] = 5
    np.save('data/tumor_enlarged.npy', matrix)
    print('体素化完成')

    matrix = np.zeros((len(x_group) + 2, len(y_group) + 2, len(z_group) + 2))
    bias_x = round((x_min6 - x_min) / step)
    bias_y = round((y_min6 - y_min) / step)
    bias_z = round((z_min6 - z_min) / step)
    for p in f:
        x = np.where(x_group6 == p[0])[0][0] + bias_x
        y = np.where(y_group6 == p[1])[0][0] + bias_y
        z = np.where(z_group6 == p[2])[0][0] + bias_z

        if(x>=matrix.shape[0] or y>=matrix.shape[1] or z>=matrix.shape[2] or x<0 or y<0 or z<0):
            continue
        matrix[x, y, z] = 5
    np.save('data/liver_cut.npy', matrix)
    print('体素化完成')



