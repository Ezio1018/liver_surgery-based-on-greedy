import math
import queue
import random
from collections import Counter
from calculate_curvature import calculate_surface_curvature
import numpy as np
import pyvista as pv
from calculate_surface import calculate_surface
from copy import deepcopy
import datetime
from skimage.morphology import skeletonize
from connectionTest import correct_flood
from first_cut import first_cut


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

def annealing(i,CURVATURE_WEIGHT, DEAD_CELL_WEIGHT, SURFACE_WEIGHT, CUT_ALIVECELL_WEIGHT, CUT_VESSEL_WEIGHT, origin_point2, origin_point3, VESSEL_FIELD_WEIGHT):

    last_cut=[]
    last_cut=first_cut(origin_point2,origin_point3)
    # last_cut=first_cut()
    # last_cut=first_cut()

    # flag=False
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

    dead3 = np.load(r'data/死掉的肝静脉骨架' + str(i) + '.npy')
    dead2 = np.load(r'data/死掉的门静脉骨架' + str(i) + '.npy')

    vessel_to_point = np.load("data/vessel_to_point.npy",allow_pickle=True).item()
    point_to_vessel = np.load("data/point_to_vessel.npy",allow_pickle=True).item()


    tumor_num = len(liver5)

    dead_cell = np.load(r'data/total_dead' + str(i) + '.npy')
    Linked_liver2 = np.load("data/Linked_liver2.npy",allow_pickle=True).item()
    Linked_liver3 = np.load("data/Linked_liver3.npy",allow_pickle=True).item()


    def initiate_Linked_liver(dead_cell):
        for p in dead_cell:
            xx = point_to_vessel.get((p[0], p[1], p[2]), 0)
            if (xx == 0):
                continue

            v2 = point_to_vessel[(p[0], p[1], p[2])][0]
            p_2 = (v2[0], v2[1], v2[2])
            Linked_liver2[p_2][1] -= 1
            next_p = Linked_liver2[p_2][0]
            while (next_p != -1):
                Linked_liver2[next_p][1] -= 1
                next_p = Linked_liver2[next_p][0]

            v3 = point_to_vessel[(p[0], p[1], p[2])][1]
            p_3 = (v3[0], v3[1], v3[2])
            Linked_liver3[p_3][1] -= 1
            next_p = Linked_liver3[p_3][0]
            while (next_p != -1):
                Linked_liver3[next_p][1] -= 1
                next_p = Linked_liver3[next_p][0]

    matrix = np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))

    matrix += liver1
    total_cell = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])

    len_total_cell = len(total_cell)
    matrix += liver2
    matrix[matrix == 3] = 2
    matrix += liver3
    matrix[matrix == 5] = 3
    matrix[matrix == 4] = 3

    matrix_copy = deepcopy(matrix)

    correct_flood(matrix, 2, origin_point2[0], origin_point2[1], origin_point2[2])
    correct_flood(matrix, 3, origin_point3[0], origin_point3[1], origin_point3[2])


    for p in liver5:
        matrix[p[0], p[1], p[2]] = 5
    for p in last_cut:
        matrix[p[0], p[1], p[2]] = 5


    for p in dead2:
        if matrix[p[0], p[1], p[2]] != 5:
            matrix[p[0], p[1], p[2]] = 6
    for p in dead3:
        if matrix[p[0], p[1], p[2]] != 5:
            matrix[p[0], p[1], p[2]] = 6
    for p in dead_cell:
        if matrix[p[0], p[1], p[2]] != 5:
            matrix[p[0], p[1], p[2]] = 6


    if(len(dead2)!=0):
        dead_cell=np.concatenate((dead2,dead_cell))
    if(len(dead3)!=0):
        dead_cell=np.concatenate((dead3,dead_cell))

    initiate_Linked_liver(dead_cell)




    l5 = reshape([np.where(matrix == 5)[0]], [np.where(matrix == 5)[1]], [np.where(matrix == 5)[2]])
    l6 = reshape([np.where(matrix == 6)[0]], [np.where(matrix == 6)[1]], [np.where(matrix == 6)[2]])

    a = [1, -1, 0, 0, 0, 0]
    b = [0, 0, 1, -1, 0, 0]
    c = [0, 0, 0, 0, 1, -1]

    def check_point(sur):
        candidate_point1 = []
        surface=[]
        def check(x, y, z):
            if matrix[x, y, z] in [0, 5]:
                return False
            else:
                return True

        for p in sur:
            check1 = [check(p[0] + 1, p[1], p[2]), check(p[0] - 1, p[1], p[2]), check(p[0], p[1] + 1, p[2]),
                      check(p[0], p[1] - 1, p[2]), check(p[0], p[1], p[2] + 1), check(p[0], p[1], p[2] - 1)]
            flag=0
            for i in range(len(check1)):
                if check1[i]:
                    flag=1
                    x_p = p[0] + a[i]
                    y_p = p[1] + b[i]
                    z_p = p[2] + c[i]
                    if [x_p, y_p, z_p] not in candidate_point1:
                        candidate_point1.append([x_p, y_p, z_p])
            if(flag==1):
                surface.append(p)
        return candidate_point1,surface

    def flood(x, y, z, value, point):
        matrix1 = deepcopy(matrix)
        matrix1[point[0],point[1],point[2]]=10
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

    def update_turned_points(p, len_live_cell,can,point_info,score_list):
        if(matrix[p[0],p[1],p[2]]==2):
            l=flood(origin_point2[0],origin_point2[1],origin_point2[2],2,p)
        elif(matrix[p[0],p[1],p[2]]==3):
            l=flood(origin_point3[0],origin_point3[1],origin_point3[2],3,p)
        else:
            return len_live_cell

        deads=[]

        for point in l:
            point_list=vessel_to_point.get((point[0],point[1],point[2]),[])
            for liver_point in point_list:
                if(matrix[liver_point[0],liver_point[1],liver_point[2]]==1):
                    matrix[liver_point[0],liver_point[1],liver_point[2]]=6
                    deads.append(liver_point)
        initiate_Linked_liver(deads)
        len_live_cell-=len(deads)
        for i in range(len(can)):
            point = can[i]
            if (matrix[point[0], point[1], point[2]] == 2):
                point_info[i][4] = Linked_liver2[point[0], point[1], point[2]][1]
                score_list[i] = score(point_info[i])

            if (matrix[point[0], point[1], point[2]] == 3):
                point_info[i][4] = Linked_liver3[point[0], point[1], point[2]][1]
                score_list[i] = score(point_info[i])

        return len_live_cell

    def score(p1):

        dead_c = 0
        count_alive = 0
        cut_vessel_and_turned_points = 0
        vessel_field_force = 0
        curvature_score = 0
        vessel_or_not = 0

        if p1[1] == 1:
            count_alive = 1
        if p1[1] == 6:
            dead_c = 1
        if p1[1] in [2,3]:
            cut_vessel_and_turned_points =p1[4]
            vessel_or_not=250
            # print('cut_vessel_and_turned_points=%f' % (cut_vessel_and_turned_points * CUT_VESSEL_WEIGHT))

        if (p1[1] in [1, 6]):
            vessel_field_force = vessel_field(p1[-1])
        before,after=p1[2],p1[3]
        curvature_score=before-after
        # if(vessel_field_force<100):
        #     vessel_field_force=0

        # before_distance = abs(before-TAGERT_CURVATURE)
        # after_distance = abs(after-TAGERT_CURVATURE)
        # curvature_score = before_distance*(after_distance-before_distance)
         # print(p1[1])
        # print('curvature_score=%f' % (curvature_score * CURVATURE_WEIGHT))
        # print('dead_c=%.3f' % (DEAD_CELL_WEIGHT * dead_c))
        # print('count_alive=%f' % (CUT_ALIVECELL_WEIGHT * count_alive))
        # print('surface_score=%f' % (SURFACE_WEIGHT * p1[0]))
        # if(p1[1] in [2,3]):
        #     print('cut_vessel_and_turned_points=%f' % (cut_vessel_and_turned_points * CUT_VESSEL_WEIGHT))
        # print('vessel_field_force=%f' % (vessel_field_force * VESSEL_FIELD_WEIGHT))
        # print()
        # 计算点的分数
        point_score = curvature_score * CURVATURE_WEIGHT+ DEAD_CELL_WEIGHT * dead_c \
                      + CUT_ALIVECELL_WEIGHT * count_alive + SURFACE_WEIGHT * p1[0] \
                      + cut_vessel_and_turned_points * CUT_VESSEL_WEIGHT + vessel_field_force * VESSEL_FIELD_WEIGHT\
                      + vessel_or_not
        return point_score

    def twenty_six_neighbor(matrix, p):
        def check(x, y, z):
            if matrix[x, y, z] == 5 or matrix[x, y, z] == 0:
                return 1
            else:
                return 0

        x_a = []
        y_b = []
        z_c = []
        for a in range(-1, 2, 1):
            for b in range(-1, 2, 1):
                for c in range(-1, 2, 1):
                    x_a.append(a)
                    y_b.append(b)
                    z_c.append(c)
        check1 = []
        for a in range(-1, 2, 1):
            for b in range(-1, 2, 1):
                for c in range(-1, 2, 1):
                    check1.append(check(p[0] + a, p[1] + b, p[2] + c))
        num = sum(check1)
        if (num >= 3):
            return True


    def search(sur_mat,candidate_point1, score_points, p1, point_info):
        def check2(x, y, z):
            if matrix[x, y, z] in [0, 5]:
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
                    if(twenty_six_neighbor(matrix,[x_p, y_p, z_p])):
                        exit_points.append([x_p, y_p, z_p])
                else:
                    # pass
                    p=p[0]
                    point_info[p][0]-=2
                    # pp = candidate_point1[p]
                    # before, after = difference_curvature(sur_mat, pp)
                    # point_info[p][2] = before
                    # point_info[p][3] = after
                    # score_points[p] = score(point_info[p])

        x_p = p1[0]
        y_p = p1[1]
        z_p = p1[2]
        distance = 1
        x_min = x_p - distance
        x_max = x_p + distance
        y_min = y_p - distance
        y_max = y_p + distance
        z_min = z_p - distance
        z_max = z_p + distance
        for i in range(x_min,x_max):
            for j in range(y_min,y_max):
                for k in range(z_min,z_max):
                    l = list(np.argwhere(candidate_point1 == [i, j, k])[:, 0])
                    d = dict(Counter(l))
                    p = [key for key, value in d.items() if value == 3]
                    if(p):
                        p=p[0]
                        pp = candidate_point1[p]
                        before, after = difference_curvature(sur_mat, pp)
                        point_info[p][2] = before
                        point_info[p][3] = after
                        score_points[p] = score(point_info[p])

        return exit_points, score_points

    def point_state(can,sur):
        point_surface = []
        def check(x, y, z):
            if matrix[x, y, z] in [0, 5]:
                return False
            else:
                return True

        for p in can:
            # print(matrix[p[0], p[1], p[2]])
            b,a=difference_curvature(sur, p)

            count = 0
            check1 = [check(p[0] + 1, p[1], p[2]), check(p[0] - 1, p[1], p[2]), check(p[0], p[1] + 1, p[2]),
                      check(p[0], p[1] - 1, p[2]), check(p[0], p[1], p[2] + 1), check(p[0], p[1], p[2] - 1)]
            for i in range(len(check1)):
                if check1[i]:
                    continue

                count -= 2
            count += 6
            if matrix[p[0], p[1], p[2]] == 1:
                point_surface.append([count, 1, b, a, p])
            elif matrix[p[0], p[1], p[2]] == 6:
                point_surface.append([count, 6, b, a, p])
            elif matrix[p[0], p[1], p[2]] == 2:
                point_surface.append([count, 2, b, a, Linked_liver2[p[0],p[1],p[2]][1], p])
            elif matrix[p[0], p[1], p[2]] == 3:
                point_surface.append([count, 3, b, a, Linked_liver3[p[0],p[1],p[2]][1], p])

        return point_surface

    def check_sur(p):
        def check(x, y, z):
            if matrix[x, y, z] in [0, 5]:
                return False
            else:
                return True
        count = 0
        check1 = [check(p[0] + 1, p[1], p[2]), check(p[0] - 1, p[1], p[2]), check(p[0], p[1] + 1, p[2]),
                  check(p[0], p[1] - 1, p[2]), check(p[0], p[1], p[2] + 1), check(p[0], p[1], p[2] - 1)]
        for i in range(len(check1)):
            if check1[i]:
                continue
            count -= 2
        count += 6
        return count

    def difference_curvature(initial_surface_mat,p):
        def check_neighbor(x, y, z):
            if initial_surface_mat[x, y, z] != 1:
                return False
            else:
                return True

        before,length=curvature(p,initial_surface_mat)
        mat=deepcopy(initial_surface_mat)

        check_1 = [check_neighbor(p[0] + 1, p[1], p[2]), check_neighbor(p[0] - 1, p[1], p[2]), check_neighbor(p[0], p[1] + 1, p[2]),
                   check_neighbor(p[0], p[1] - 1, p[2]), check_neighbor(p[0], p[1], p[2] + 1), check_neighbor(p[0], p[1], p[2] - 1)]

        for i in range(len(check_1)):
            x_p = p[0] + a[i]
            y_p = p[1] + b[i]
            z_p = p[2] + c[i]
            if check_1[i]:
                sur = check_sur([x_p,y_p,z_p])
                if(sur == -4):
                    mat[x_p,y_p,z_p]=0
        mat[p[0],p[1],p[2]]=1
        after,length=curvature(p,mat)

        if(length<=25):
            before, after = -500, 500

        return before,after

    def update_surface(initial_surface_mat,p):
        def check_neighbor(x, y, z):
            if initial_surface_mat[x, y, z] != 1:
                return False
            else:
                return True

        check_1 = [check_neighbor(p[0] + 1, p[1], p[2]), check_neighbor(p[0] - 1, p[1], p[2]), check_neighbor(p[0], p[1] + 1, p[2]),
                   check_neighbor(p[0], p[1] - 1, p[2]), check_neighbor(p[0], p[1], p[2] + 1), check_neighbor(p[0], p[1], p[2] - 1)]

        for i in range(len(check_1)):
            x_p = p[0] + a[i]
            y_p = p[1] + b[i]
            z_p = p[2] + c[i]
            if check_1[i]:
                sur = check_sur([x_p,y_p,z_p])
                if(sur == -6):
                    initial_surface_mat[x_p,y_p,z_p]=0
        initial_surface_mat[p[0],p[1],p[2]]=1

        return initial_surface_mat



    def curvature(test_point,surface):
        # print(test_point)
        x_p = test_point[0]
        y_p = test_point[1]
        z_p = test_point[2]
        distance = 3
        x_min = x_p - distance
        x_max = x_p + distance
        y_min = y_p - distance
        y_max = y_p + distance
        z_min = z_p - distance
        z_max = z_p + distance

        temp = surface[x_min:x_max, y_min:y_max, z_min:z_max]
        l = reshape([np.where(temp == 1)[0]], [np.where(temp == 1)[1]],
                    [np.where(temp == 1)[2]])


        return calculate_surface_curvature(l),len(l)

    def update_Linked_liver(p,can,score_list,point_info):
        if(matrix[p[0],p[1],p[2]]==1):
            m2=point_to_vessel[(p[0],p[1],p[2])][0]
            m2=(m2[0],m2[1],m2[2])
            m3=point_to_vessel[(p[0],p[1],p[2])][1]
            m3=(m3[0],m3[1],m3[2])

            while(m2!=-1):
                # print("liver2:{}".format(m2))
                Linked_liver2[(m2[0],m2[1],m2[2])][1]-=1
                m2=Linked_liver2[(m2[0],m2[1],m2[2])][0]
                if(m2!=-1):
                    m2=(m2[0],m2[1],m2[2])

            while(m3!=-1):
                Linked_liver3[(m3[0],m3[1],m3[2])][1]-=1
                m3=Linked_liver3[(m3[0],m3[1],m3[2])][0]
                if(m3!=-1):
                    m3=(m3[0],m3[1],m3[2])

            for i in range(len(can)):
                point=can[i]
                if(matrix[point[0],point[1],point[2]] == 2):
                    point_info[i][4]=Linked_liver2[point[0],point[1],point[2]][1]
                    score_list[i]=score(point_info[i])

                if(matrix[point[0],point[1],point[2]] == 3):
                    point_info[i][4]=Linked_liver3[point[0],point[1],point[2]][1]
                    score_list[i]=score(point_info[i])


    def train():

        start = datetime.datetime.now()
        global res, candidate_point, point, score_point, point_index

        len_live_cell = len_total_cell - len(l6) - len(l5)
        Proportion = round(len_live_cell / len_total_cell, 2)
        print(Proportion)
        done = (1-Proportion)*0.8*len_total_cell-len(last_cut)


        print("活细胞体积:", len_live_cell)
        print("总体积体积:", len_total_cell)
        print("活细胞占比:", Proportion)

        surface = calculate_surface(matrix, 5)
        candidate_point, surface = check_point(surface)
        # plot1(surface)
        surface_mat=np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))
        for p in surface:
            surface_mat[p[0],p[1],p[2]]=1
        point_info = point_state(candidate_point,surface_mat)
        counter=tumor_num
        while True:
            if counter == tumor_num:

                score_point = []
                for p in point_info:
                    score_point.append(score(p))
                score_point = np.array(score_point)
                # plot()
            else:
                score_point = np.delete(score_point, point_index)
                candidate_point = np.delete(candidate_point, point_index, axis=0)
                point_info.pop(point_index)
                exit_points, score_point = search(surface_mat, candidate_point, score_point, point, point_info)

                if exit_points:
                    exit_points_surface = point_state(exit_points,surface_mat)
                    for p1 in exit_points_surface:
                        # print("exit{}".format(p1))
                        point_info.append(p1)
                        score_point = np.append(score_point, score(p1))
                    for p2 in exit_points:
                        candidate_point = np.concatenate((candidate_point, [p2]))


            if  counter > done :
                print('优化结束!')
                R_cut = reshape([np.where(matrix == 5)[0]], [np.where(matrix == 5)[1]], [np.where(matrix == 5)[2]])
                m_res = flood1(origin_point2[0], origin_point2[1], origin_point2[2], 2, matrix)
                g_res = flood1(origin_point3[0], origin_point3[1], origin_point3[2], 3, matrix)
                m_res = list(m_res)
                g_res = list(g_res)
                for point in R_cut:
                    if matrix_copy[point[0], point[1], point[2]] == 2:
                        m_res.append(point)
                    if matrix_copy[point[0], point[1], point[2]] == 3:
                        g_res.append(point)
                ssss = reshape([np.where(surface_mat == 1)[0]], [np.where(surface_mat == 1)[1]], [np.where(surface_mat == 1)[2]])

                np.save('optdata/R_data' + str(i) + '.npy', candidate_point)
                np.save('optdata/R_cut' + str(i) + '.npy', R_cut)
                np.save('optdata/m_res' + str(i) + '.npy', m_res)
                np.save('optdata/g_res' + str(i) + '.npy', g_res)
                np.save('optdata/surface' + str(i) + '.npy', ssss)

                print("活细胞占比:", round(len_live_cell / len_total_cell, 2))
                end = datetime.datetime.now()

                # print('Running time: %s Seconds' % (end - start))

                plot()
                break


            point_index = np.argmax(score_point)
            point = candidate_point[point_index]



            counter += 1
            if(matrix[point[0], point[1], point[2]]==1):
                len_live_cell -= 1
            update_Linked_liver(point,candidate_point,score_point,point_info)
            len_live_cell = update_turned_points(point,len_live_cell,candidate_point,point_info,score_point)
            matrix[point[0], point[1], point[2]] = 5

            surface_mat=update_surface(surface_mat,point)

            if counter % 1000 == 1:
                print('第', counter, '次添加点')
                print("活细胞占比:", round(len_live_cell / len_total_cell, 2))



        # before,after=difference_curvature(surface_mat,[63,41,72])

    def vessel_field(p):
        if(point_to_vessel.get((p[0],p[1],p[2]),0)==0):
            return 0
        m2 = point_to_vessel[(p[0],p[1],p[2])][0]
        m3 = point_to_vessel[(p[0],p[1],p[2])][1]
        l2 = Linked_liver2[(m2[0],m2[1],m2[2])][1]
        l3 = Linked_liver3[(m3[0],m3[1],m3[2])][1]

        return min(l2,l3)

    def plot():
        l1 = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])
        # l5 = reshape([np.where(matrix == 5)[0]], [np.where(matrix == 5)[1]], [np.where(matrix == 5)[2]])
        l6 = reshape([np.where(matrix == 6)[0]], [np.where(matrix == 6)[1]], [np.where(matrix == 6)[2]])
        l2 = reshape([np.where(matrix == 2)[0]], [np.where(matrix == 2)[1]], [np.where(matrix == 2)[2]])
        l3 = reshape([np.where(matrix == 3)[0]], [np.where(matrix == 3)[1]], [np.where(matrix == 3)[2]])

        # print(l1)
        p = pv.Plotter()
        p.set_background("white")
        p.add_mesh(pv.PolyData(l1), color='blue', opacity=0.03)
        # p.add_mesh(pv.PolyData(l6), color='yellow', opacity=0.03)
        p.add_mesh(pv.PolyData(l2), color='green', render_points_as_spheres=True)
        p.add_mesh(pv.PolyData(l3), color='purple', render_points_as_spheres=True)
        # p.add_mesh(pv.PolyData(l4), color='orange', render_points_as_spheres=True)
        # p.add_mesh(pv.PolyData(l5), color='black', opacity=
        # p.add_mesh(pv.PolyData(m_res), color='blue', opacity=0.4)
        # p.add_mesh(pv.PolyData(g_res), color='orange', opacity=0.4)

        if(len(l6!=0)):
            p.add_mesh(pv.PolyData(l6), color='yellow', opacity=0.04)
        p.add_mesh(pv.PolyData(candidate_point), color='red', render_points_as_spheres=True)
        # p.add_mesh(pv.PolyData(candidate_point), color='blue', render_points_as_spheres=True, opacity=0.1)
        p.show()

    train()
#
# CURVATURE_WEIGHT = 0
# DEAD_CELL_WEIGHT = 50 #死细胞的权重
# CUT_ALIVECELL_WEIGHT = -50#活细胞的权重
# SURFACE_WEIGHT = -10 #表面积的权重
# CUT_VESSEL_WEIGHT = -0.5
# VESSEL_FIELD_WEIGHT = -5
#
# i=1
#
# # 门静脉起始填充点(红色)
# x_l1 = 103
# y_l1 = 52
# z_l1 = 92
# # 肝静脉起始填充点(蓝色)
# x_l2 = 92
# y_l2 = 30
# z_l2 = 59
# annealing(i,CURVATURE_WEIGHT,DEAD_CELL_WEIGHT, SURFACE_WEIGHT, CUT_ALIVECELL_WEIGHT, CUT_VESSEL_WEIGHT, [x_l1,y_l1,z_l1], [x_l2,y_l2,z_l2], VESSEL_FIELD_WEIGHT)