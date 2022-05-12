import math
import random
from collections import Counter
from calculate_curvature import calculate_surface_curvature
import numpy as np
import pyvista as pv
from calculate_surface import calculate_surface
from copy import deepcopy

i = 1



def annealing(i, DEAD_CELL_WEIGHT, SURFACE_WEIGHT, CUT_ALIVECELL_WEIGHT,flag = True,cut=[]):

    # flag=False
    liver1 = np.load(r'data/肝实质.npy')
    liver2 = np.load(r'data/门静脉.npy')
    liver3 = np.load(r'data/肝静脉.npy')
    # liver4 = np.load(r'file/动脉.npy')
    liver5 = np.load(r'data/随机占位' + str(i) + '.npy')
    dead3 = np.load(r'data/死掉的肝静脉' + str(i) + '.npy')
    dead2 = np.load(r'data/死掉的门静脉' + str(i) + '.npy')
    dead_cell = np.load(r'data/total_dead' + str(i) + '.npy')
    dead_cell=np.concatenate((dead2,dead_cell))
    dead_cell=np.concatenate((dead3,dead_cell))
    matrix = np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))

    def reshape(a, b, c):
        a = np.array(a).T
        b = np.array(b).T
        c = np.array(c).T
        m = np.concatenate((a, b, c), axis=1)
        return m

    matrix += liver1
    total_cell = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])

    len_total_cell = len(total_cell)
    matrix += liver2
    matrix[matrix == 3] = 2
    matrix += liver3
    matrix[matrix == 5] = 3
    matrix[matrix == 4] = 3

    for p in liver5:
        matrix[p[0], p[1], p[2]] = 5
    # matrix[matrix == 7] = 5
    # matrix[matrix == 8] = 5
    # matrix[matrix == 9] = 5

    for p in dead2:
        if matrix[p[0], p[1], p[2]] != 5:
            matrix[p[0], p[1], p[2]] = 6
    for p in dead3:
        if matrix[p[0], p[1], p[2]] != 5:
            matrix[p[0], p[1], p[2]] = 6
    for p in dead_cell:
        if matrix[p[0], p[1], p[2]] != 5:
            matrix[p[0], p[1], p[2]] = 6

    for p in cut:
        matrix[p[0], p[1], p[2]] = 5

    def reshape(a, b, c):
        a = np.array(a).T
        b = np.array(b).T
        c = np.array(c).T
        m = np.concatenate((a, b, c), axis=1)
        return m

    l1 = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])
    # l2 = reshape([np.where(matrix == 2)[0]], [np.where(matrix == 2)[1]], [np.where(matrix == 2)[2]])
    # l3 = reshape([np.where(matrix == 3)[0]], [np.where(matrix == 3)[1]], [np.where(matrix == 3)[2]])
    # l4 = reshape([np.where(matrix == 4)[0]], [np.where(matrix == 4)[1]], [np.where(matrix == 4)[2]])
    l5 = reshape([np.where(matrix == 5)[0]], [np.where(matrix == 5)[1]], [np.where(matrix == 5)[2]])
    l6 = reshape([np.where(matrix == 6)[0]], [np.where(matrix == 6)[1]], [np.where(matrix == 6)[2]])
    len_live_cell = len_total_cell - len(l6) - len(l5)
    Proportion = round(len_live_cell / len_total_cell, 2)
    done = Proportion * 0.85 + 0.05  # 0.85为切除点平衡权重
    print(done)
    if done < 0.4:
        done = 0.4
    print("活细胞体积:", len_live_cell)
    print("总体积体积:", len_total_cell)
    print("活细胞占比:", Proportion)

    a = [1, -1, 0, 0, 0, 0]
    b = [0, 0, 1, -1, 0, 0]
    c = [0, 0, 0, 0, 1, -1]

    def check_point(sur):
        candidate_point1 = []

        def check(x, y, z):
            if matrix[x, y, z] in [0, 5, 6]:
                return False
            else:
                return True

        for p in sur:
            check1 = [check(p[0] + 1, p[1], p[2]), check(p[0] - 1, p[1], p[2]), check(p[0], p[1] + 1, p[2]),
                      check(p[0], p[1] - 1, p[2]), check(p[0], p[1], p[2] + 1), check(p[0], p[1], p[2] - 1)]
            for i in range(len(check1)):
                if check1[i]:
                    x_p = p[0] + a[i]
                    y_p = p[1] + b[i]
                    z_p = p[2] + c[i]
                    if [x_p, y_p, z_p] not in candidate_point1:
                        candidate_point1.append([x_p, y_p, z_p])
        return candidate_point1



    def score(p1):

        dead_c = 0
        count_alive = 0
        if p1[1] in [1]:
            count_alive = 1
        if p1[1] == 6:
            dead_c = 1
        # 计算点的分数
        point_score = DEAD_CELL_WEIGHT * dead_c + CUT_ALIVECELL_WEIGHT * count_alive + SURFACE_WEIGHT * p1[0]
        return point_score

    def search(candidate_point1, score_points, p1):
        def check2(x, y, z):
            if matrix[x, y, z] in [0, 5, 6]:
                return False
            else:
                return True

        exit_points = []
        check_1 = [check2(p1[0] + 1, p1[1], p1[2]), check2(p1[0] - 1, p1[1], p1[2]), check2(p1[0], p1[1] + 1, p1[2]),
                   check2(p1[0], p1[1] - 1, p1[2]), check2(p1[0], p1[1], p1[2] + 1), check2(p1[0], p1[1], p1[2] - 1)]
        for i in range(len(check_1)):
            x_p = p1[0] + a[i]
            y_p = p1[1] + b[i]
            z_p = p1[2] + c[i]

            if check_1[i]:

                l = list(np.argwhere(candidate_point1 == [x_p, y_p, z_p])[:, 0])
                d = dict(Counter(l))
                p = [key for key, value in d.items() if value == 3]
                if not p:
                    exit_points.append([x_p, y_p, z_p])
                else:
                    p = p[0]
                    score_points[p] -= 2 * SURFACE_WEIGHT
        return exit_points, score_points

    def six_surface(sur):
        point_surface = []
        def check(x, y, z):
            if matrix[x, y, z] in [0, 5, 6]:
                return False
            else:
                return True

        for p in sur:
            # print(matrix[p[0], p[1], p[2]])
            count = 0
            check1 = [check(p[0] + 1, p[1], p[2]), check(p[0] - 1, p[1], p[2]), check(p[0], p[1] + 1, p[2]),
                      check(p[0], p[1] - 1, p[2]), check(p[0], p[1], p[2] + 1), check(p[0], p[1], p[2] - 1)]
            for i in range(len(check1)):
                if check1[i]:
                    continue

                count -= 2
            count += 6
            if matrix[p[0], p[1], p[2]] in [1,2,3]:
                point_surface.append([count, 1])
            elif matrix[p[0], p[1], p[2]] in [5,6]:
                point_surface.append([count, 6])
        return point_surface

    def Judge(score, T):
        if score < 0:
            return 1
        else:
            probability = math.exp(-score / T)
            if probability > random.random():
                return 1
            else:
                return 0

    def difference_curvature(initial_surface_mat,p):
        def check_neighbor(x, y, z):
            if initial_surface_mat[x, y, z] !=1:
                return False
            else:
                return True

        before=curvature(p,initial_surface_mat)
        mat=deepcopy(initial_surface_mat)

        check_1 = [check_neighbor(p[0] + 1, p[1], p[2]), check_neighbor(p[0] - 1, p[1], p[2]), check_neighbor(p[0], p[1] + 1, p[2]),
                   check_neighbor(p[0], p[1] - 1, p[2]), check_neighbor(p[0], p[1], p[2] + 1), check_neighbor(p[0], p[1], p[2] - 1)]

        for i in range(len(check_1)):
            x_p = p[0] + a[i]
            y_p = p[1] + b[i]
            z_p = p[2] + c[i]
            if check_1[i]:
                # print(six_surface([[x_p,y_p,z_p]]))
                sur=six_surface([[x_p,y_p,z_p]])[0][0]
                if(sur==-4):
                    mat[x_p,y_p,z_p]=0
        mat[p[0],p[1],p[2]]=1
        after=curvature(p,mat)

        return before,after




    def curvature(test_point,surface):
        x_p = test_point[0]
        y_p = test_point[1]
        z_p = test_point[2]
        distance = 2
        x_min = x_p - distance
        x_max = x_p + distance
        y_min = y_p - distance
        y_max = y_p + distance
        z_min = z_p - distance
        z_max = z_p + distance

        temp = surface[x_min:x_max, y_min:y_max, z_min:z_max]
        l = reshape([np.where(temp == 1)[0]], [np.where(temp == 1)[1]],
                    [np.where(temp == 1)[2]])
        plot(l)
        return calculate_surface_curvature(l)




    def train(dead_cell):
        global res, candidate_point, point, score_point, point_index
        dead_cell=list(dead_cell)
        res = list(cut[:])
        m_res = []
        g_res = []
        tmp = 1000
        alpha = 0.98
        # len_live_cell = len_total_cell - len(l6) - len(l5)
        # surface = np.load(r'surface_dead.npy')
        surface = calculate_surface(matrix, 5)
        if flag:
            dead5 = calculate_surface(matrix, 6)
            surface = np.concatenate((surface, dead5))

        total = 1
        counter = 0
        candidate_point = check_point(surface)
        # print(candidate_point[100])
        surface_mat=np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))
        # plot(candidate_point)
        for p in surface:
            surface_mat[p[0],p[1],p[2]]=1
        point_surface = six_surface(candidate_point)
        # before,after=difference_curvature(surface_mat,[70,45,64])

            # print(p)
        before,after=difference_curvature(surface_mat,[63,41,72])


    def plot(surface):
        l1 = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])
        l6 = reshape([np.where(matrix == 6)[0]], [np.where(matrix == 6)[1]], [np.where(matrix == 6)[2]])
        # print(l1)
        def show_cor(cor):
            print(cor)
        p = pv.Plotter()
        # p.enable_point_picking(callback=show_cor)
        p.show_grid()
        for i in range(len(surface)):
            t1 = pv.Cube(surface[i])
            p.add_mesh(t1, color='red', opacity=1, show_edges=True)
        # for i in range(len(can)):
        #     t1 = pv.Cube(can[i])
        #     p.add_mesh(t1, color='yellow', opacity=1, show_edges=True)
        # p.add_mesh(pv.PolyData(surface), color='blue', opacity=1)
        p.show()

    train(dead_cell)

DEAD_CELL_WEIGHT = 1.1 #死细胞的权重
CUT_ALIVECELL_WEIGHT = -0.5 #活细胞的权重
SURFACE_WEIGHT = 2 #表面积的权重
annealing(i,DEAD_CELL_WEIGHT, SURFACE_WEIGHT, CUT_ALIVECELL_WEIGHT,True)