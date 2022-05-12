import queue
from copy import deepcopy

import  numpy as np
from skimage.morphology import skeletonize

from calculate_surface import calculate_surface
from connectionTest import correct_flood
from main5_dataprocess import process

def reshape(a, b, c):
    a = np.array(a).T
    b = np.array(b).T
    c = np.array(c).T
    m = np.concatenate((a, b, c), axis=1)
    return m

def flood1(x, y, z, value,matrix):
    matrix1 = deepcopy(matrix)

    def check(x, y, z):
        if matrix1[x, y, z] == value:
            return False
        else:
            return True

    q = queue.Queue()
    if check(x, y, z):
        return
    q.put([x, y, z])
    matrix1[x, y, z] = 10
    x_a = []
    y_b = []
    z_c = []
    for a in range(-1, 2, 1):
        for b in range(-1, 2, 1):
            for c in range(-1, 2, 1):
                x_a.append(a)
                y_b.append(b)
                z_c.append(c)
    count = 0
    while not q.empty():
        count += 1
        p = q.get()
        check1 = []
        for a in range(-1, 2, 1):
            for b in range(-1, 2, 1):
                for c in range(-1, 2, 1):
                    check1.append(check(p[0] + a, p[1] + b, p[2] + c))
        for i in range(len(check1)):
            if check1[i]:
                continue

            x_p = p[0] + x_a[i]
            y_p = p[1] + y_b[i]
            z_p = p[2] + z_c[i]

            if matrix1[x_p, y_p, z_p] == value:
                matrix1[x_p, y_p, z_p] = 10
                q.put([x_p, y_p, z_p])

    l = reshape([np.where(matrix1 == value)[0]], [np.where(matrix1 == value)[1]],
                [np.where(matrix1 == value)[2]])
    return l

def weipu_score(origin_point2,origin_point3):
    x_l1, y_l1, z_l1 = origin_point2[0], origin_point2[1], origin_point2[2]
    x_l2, y_l2, z_l2 = origin_point3[0], origin_point3[1], origin_point3[2]
    a = [1, -1, 0, 0, 0, 0]
    b = [0, 0, 1, -1, 0, 0]
    c = [0, 0, 0, 0, 1, -1]
    dead=np.load("data/liver_cut.npy")
    liver1 = np.load(r'data/liver.npy')
    liver2 = np.load('data/m_vessel.npy')
    liver2[liver2 == 2] = 1
    liver3 = np.load('data/g_vessel.npy')
    liver3[liver3 == 3] = 1
    liver5 = np.load("data/tumor_random1.npy")

    liver3 = skeletonize(liver3)
    liver3[liver3 == 255] = 3
    liver2 = skeletonize(liver2)
    liver2[liver2 == 255] = 2


    def check_point(matrix, sur):
        candidate_point1 = []
        surface = []

        def check(x, y, z):
            if (x >= liver1.shape[0] or x < 0 or y >= liver1.shape[1] or y < 0 or z >= liver1.shape[2] or z < 0):
                return False
            if matrix[x, y, z] in [0, 5]:
                return False
            else:
                return True

        for p in sur:
            check1 = [check(p[0] + 1, p[1], p[2]), check(p[0] - 1, p[1], p[2]), check(p[0], p[1] + 1, p[2]),
                      check(p[0], p[1] - 1, p[2]), check(p[0], p[1], p[2] + 1), check(p[0], p[1], p[2] - 1)]
            flag = 0
            for i in range(len(check1)):
                if check1[i]:
                    flag = 1
                    x_p = p[0] + a[i]
                    y_p = p[1] + b[i]
                    z_p = p[2] + c[i]
                    if [x_p, y_p, z_p] not in candidate_point1:
                        candidate_point1.append([x_p, y_p, z_p])
            if (flag == 1):
                surface.append(p)
        return candidate_point1, surface



    matrix = np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))

    matrix += liver1
    total_cell = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])

    len_total_cell = len(total_cell)
    matrix += liver2
    matrix[matrix == 3] = 2
    matrix += liver3
    matrix[matrix == 5] = 3
    matrix[matrix == 4] = 3

    correct_flood(matrix, 2, origin_point2[0], origin_point2[1], origin_point2[2])
    correct_flood(matrix, 3, origin_point3[0], origin_point3[1], origin_point3[2])


    R_cut = reshape([np.where(dead == 5)[0]], [np.where(dead == 5)[1]], [np.where(dead == 5)[2]])

    for p in R_cut:
        matrix[p[0],p[1],p[2]]=5

    m_res = flood1(origin_point2[0], origin_point2[1], origin_point2[2], 2, matrix)
    g_res = flood1(origin_point3[0], origin_point3[1], origin_point3[2], 3, matrix)
    m_res=list(m_res)
    g_res=list(g_res)

    for point in R_cut:
        if matrix[point[0], point[1], point[2]] == 2:
            m_res.append(point)
        if matrix[point[0], point[1], point[2]] == 3:
            g_res.append(point)

    surface = calculate_surface(matrix, 5)
    candidate_point, surface = check_point(matrix,surface)

    np.save("optdata/R_data1.npy",surface)
    np.save("optdata/R_cut1.npy",R_cut)
    np.save('optdata/m_res1.npy', m_res)
    np.save('optdata/g_res1.npy', g_res)

    process(1)

