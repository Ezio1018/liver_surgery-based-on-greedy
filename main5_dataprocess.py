import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from calculate_surface import calculate_surface
import open3d as o3d

from point_to_pcd import points2pcd
from total_curvature import caculate_surface_curvature


def process(i):
    liver1 = np.load('data/liver.npy')
    liver2 = np.load('data/m_vessel.npy')
    liver3 = np.load('data/g_vessel.npy')
    blood1 = np.load(r'data/淤血区域' + str(i) + '.npy')
    blood2 = np.load(r'data/缺血区域' + str(i) + '.npy')
    point = np.load('optdata/R_data' + str(i) + '.npy')
    point_cut = np.load('optdata/R_cut' + str(i) + '.npy')
    tumour = np.load('data/tumor.npy')

    skeleton_to_vessel2 = np.load("data/skeleton_to_vessel2_dict.npy", allow_pickle=True).item()
    skeleton_to_vessel3 = np.load("data/skeleton_to_vessel3_dict.npy", allow_pickle=True).item()

    def reshape(a, b, c):
        a = np.array(a).T
        b = np.array(b).T
        c = np.array(c).T
        m = np.concatenate((a, b, c), axis=1)
        return m

    liver3[liver3==3]=1
    liver2[liver2==2]=1
    skeleton3 = skeletonize(liver3)
    skeleton3[skeleton3 == 255] = 3
    skeleton2 = skeletonize(liver2)
    skeleton2[skeleton2 == 255] = 2
    skeleton2[29, 6, 12] = 0
    skeleton2[29, 5, 12] = 2
    skeleton2[29, 5, 13] = 2
    skeleton2[29, 6, 13] = 2

    liver2 = np.load('data/m_vessel.npy')
    liver3 = np.load('data/g_vessel.npy')




    vessel_to_point = np.load('data/vessel_to_point.npy', allow_pickle=True).item()
    m_res = np.load("optdata/m_res1.npy")
    g_res = np.load("optdata/g_res1.npy")
    R_cut = np.load("optdata/R_cut1.npy")
    m_res_cut = []
    for p in m_res:
        p_list = skeleton_to_vessel2[(p[0], p[1], p[2])]
        if (len(p_list) == 0):
            continue
        if (len(m_res_cut) != 0):
            m_res_cut = np.vstack((m_res_cut, p_list))
        else:
            m_res_cut = p_list
    g_res_cut = []
    for p in g_res:
        p_list = skeleton_to_vessel3[(p[0], p[1], p[2])]
        if (len(p_list) == 0):
            continue
        if (len(g_res_cut) != 0):
            g_res_cut = np.vstack((g_res_cut, p_list))
        else:
            g_res_cut = p_list


    def search(l):
        total_dead = np.array([])

        for p in l:
            # print(p)
            dead = np.array(vessel_to_point.get((p[0], p[1], p[2]), []))
            if (len(dead)==0):
                continue
            if (len(total_dead) == 0):
                total_dead = dead
            else:
                total_dead = np.concatenate((total_dead, dead))
        return total_dead

    a1 = search(m_res)
    a2 = search(g_res)
    cut_rows = R_cut.view([('', R_cut.dtype)] * R_cut.shape[1])
    m_remain = 0
    g_remain = 0

    if (len(a1) != 0):
        a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
        a1_cut = np.intersect1d(a1_rows, cut_rows).view(a1.dtype).reshape(-1, a1.shape[1])
        m_remain = len(a1) - len(a1_cut)
    if (len(a2) != 0):
        a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
        a2_cut = np.intersect1d(a2_rows, cut_rows).view(a2.dtype).reshape(-1, a2.shape[1])
        g_remain = len(a2) - len(a2_cut)

    matrix = np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))
    matrix += liver1
    matrix += tumour
    matrix[matrix == 5] = 4
    total = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])
    tumour_len = len(reshape([np.where(matrix == 4)[0]], [np.where(matrix == 4)[1]], [np.where(matrix == 4)[2]]))
    len_total = len(total)





    for p in point_cut:
        if matrix[p[0], p[1], p[2]] != 5 and matrix[p[0], p[1], p[2]] != 0:
            matrix[p[0], p[1], p[2]] = 6


    for p in m_res_cut:
        if matrix[p[0], p[1], p[2]] != 5:
            matrix[p[0], p[1], p[2]] = 6
    for p in g_res_cut:
        if matrix[p[0], p[1], p[2]] != 5:
            matrix[p[0], p[1], p[2]] = 6

    l1 = calculate_surface(matrix, 1)
    l6 = calculate_surface(matrix, 6)
    np.save('res/liver_remain' + str(i) + '.npy', l1)
    np.save('res/liver_cut' + str(i) + '.npy', l6)

    l1 = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])
    l6 = reshape([np.where(matrix == 6)[0]], [np.where(matrix == 6)[1]], [np.where(matrix == 6)[2]])
    matrix += liver2
    matrix[matrix == 3] = 2
    matrix[matrix == 7] = 2
    matrix[matrix == 8] = 2
    matrix += liver3
    matrix[matrix == 4] = 3
    matrix[matrix == 8] = 3
    matrix[matrix == 9] = 3
    l2 = calculate_surface(matrix, 2)
    l3 = calculate_surface(matrix, 3)

    np.save('res/g_vessel_cut' + str(i) + '.npy', l3)
    np.save('res/m_vessel_cut' + str(i) + '.npy', l2)

    def six_surface(sur):
        sum_sur = 0

        def check(x, y, z):
            if matrix[x, y, z] in [6, 5]:
                return False
            else:
                return True

        for p in sur:
            check1 = [check(p[0] + 1, p[1], p[2]), check(p[0] - 1, p[1], p[2]), check(p[0], p[1] + 1, p[2]),
                      check(p[0], p[1] - 1, p[2]), check(p[0], p[1], p[2] + 1), check(p[0], p[1], p[2] - 1)]
            for i in range(len(check1)):
                if check1[i]:
                    continue
                sum_sur += 1.44
        return sum_sur

    surface_p = six_surface(point)
    points2pcd(point)
    pcd = o3d.io.read_point_cloud("pcd/cache.pcd")
    surface_curvature = caculate_surface_curvature(pcd, radius=5)
    mean_sur_cur = np.mean(surface_curvature)

    print('切除肝脏体积占比:', round((len(l6)) / len_total, 4) * 100, '%')
    print('肿瘤体积占比:', round((tumour_len / len_total) * 100, 4), '%')
    print('肝脏断面面积:', round(surface_p, 4), 'mm^2')
    print('起初淤血血体积占比:', round((len(blood1) / len_total) * 100, 4), '%')
    print('起初缺血血体积占比:', round((len(blood2) / len_total) * 100, 4), '%')
    print('优化后缺血体积占比:', round((m_remain / len_total) * 100, 4), '%')
    print('优化后淤血体积占比:', round((g_remain / len_total) * 100, 4), '%')

    reader = pd.read_excel('my.xlsx')
    data = [str(i), str(round((len(l6)) / len_total, 3) * 100) + "%",
            str(round((tumour_len / len_total) * 100, 4)) + "%",
            str(round(surface_p, 4)) + "mm^2"
        , str(round((len(blood1) / len_total) * 100, 4)) + "%",
            str(round((len(blood2) / len_total) * 100, 4)) + "%", str(round((m_remain / len_total) * 100, 4)) + "%"
        , str(round((g_remain / len_total) * 100, 4)) + "%", mean_sur_cur]

    reader.loc[len(reader)] = data
    writer = pd.ExcelWriter('my.xlsx')
    reader.to_excel(writer, index=False)
    writer.save()
    print('保存成功!')
