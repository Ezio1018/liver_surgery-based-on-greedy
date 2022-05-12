import numpy as np
from skimage.morphology import skeletonize
from connectionTest import correct_flood


# 门静脉起始填充点(红色)
from dead_total_version3 import dead_total

def first_cut(origin_point2, origin_point3):
    x_l1, y_l1, z_l1 = origin_point2[0], origin_point2[1], origin_point2[2]
    x_l2, y_l2, z_l2 = origin_point3[0], origin_point3[1], origin_point3[2]
    i=1
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

    point_to_vessel = np.load("data/point_to_vessel.npy",allow_pickle=True).item()
    def reshape(a, b, c):
        a = np.array(a).T
        b = np.array(b).T
        c = np.array(c).T
        m = np.concatenate((a, b, c), axis=1)
        return m

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

    initiate_Linked_liver(dead_cell)

    def vessel_field(p):
        m2 = point_to_vessel[(p[0],p[1],p[2])][0]
        m3 = point_to_vessel[(p[0],p[1],p[2])][1]
        l2 = Linked_liver2[(m2[0],m2[1],m2[2])][1]
        l3 = Linked_liver3[(m3[0],m3[1],m3[2])][1]

        return max(l2,l3)
    matrix = np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))

    matrix += liver1
    total_cell = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])

    len_total_cell = len(total_cell)
    matrix += liver2
    matrix[matrix == 3] = 2
    matrix += liver3
    matrix[matrix == 5] = 3
    matrix[matrix == 4] = 3

    correct_flood(matrix, 2, x_l1, y_l1, z_l1)
    correct_flood(matrix, 3, x_l2, y_l2, z_l2)

    for p in liver5:
        matrix[p[0], p[1], p[2]] = 5
    #
    # for p in dead2:
    #     if matrix[p[0], p[1], p[2]] != 5:
    #         matrix[p[0], p[1], p[2]] = 6
    # for p in dead3:
    #     if matrix[p[0], p[1], p[2]] != 5:
    #         matrix[p[0], p[1], p[2]] = 6
    for p in dead_cell:
        if matrix[p[0], p[1], p[2]] != 5:
            matrix[p[0], p[1], p[2]] = 6


    if(len(dead2)!=0):
        dead_cell=np.concatenate((dead2,dead_cell))
    if(len(dead3)!=0):
        dead_cell=np.concatenate((dead3,dead_cell))


    l2 = reshape([np.where(matrix == 2)[0]], [np.where(matrix == 2)[1]], [np.where(matrix == 2)[2]])
    l3 = reshape([np.where(matrix == 3)[0]], [np.where(matrix == 3)[1]], [np.where(matrix == 3)[2]])
    l6 = reshape([np.where(matrix == 6)[0]], [np.where(matrix == 6)[1]], [np.where(matrix == 6)[2]])


    a = [1, -1, 0, 0, 0, 0]
    b = [0, 0, 1, -1, 0, 0]
    c = [0, 0, 0, 0, 1, -1]

    def check_sur(p,lists):
        def check(x, y, z):
            if matrix[x, y, z] in lists:
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

    for p in l6:
        if(vessel_field(p)>len(l6)*0.6):
            matrix[p[0],p[1],p[2]]=1

    for p in l2:
        if(check_sur(p,[6,5])<=-2):
            matrix[p[0],p[1],p[2]]=6

    for p in l3:
        if(check_sur(p,[6,5])<=-2):
            matrix[p[0],p[1],p[2]]=6

    l6 = reshape([np.where(matrix == 6)[0]], [np.where(matrix == 6)[1]], [np.where(matrix == 6)[2]])
    np.save("initial_dead",l6)


    Liver_filepath_npy = 'data/liver.npy'
    artery_filepath_npy = 'data/m_vessel.npy'
    vein_filepath_npy = 'data/g_vessel.npy'
    tumour_filepath_npy = 'data/tumor_random' + str(i) + '.npy'
    dead_total(Liver_filepath_npy, artery_filepath_npy, vein_filepath_npy, tumour_filepath_npy, i,
                  x_l1, y_l1, z_l1, x_l2, y_l2, z_l2, l6)
    return l6