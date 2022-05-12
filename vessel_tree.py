
import queue
import pyvista as pv
import numpy as np
import sys
from skimage.morphology import skeletonize
from copy import deepcopy
from connectionTest import correct_flood

from sklearn.ensemble._gradient_boosting import np_float32
def tree_vessel(origin_point2,origin_point3):
    x_l1, y_l1, z_l1 = origin_point2[0], origin_point2[1], origin_point2[2]
    x_l2, y_l2, z_l2 = origin_point3[0], origin_point3[1], origin_point3[2]
    index=np.load("data/index_skeleton.npy")
    print(index.shape)
    string = ''
    liver1 = np.load('data' + string + '/liver.npy')
    liver2 = np.load('data' + string + '/m_vessel.npy')
    liver2[liver2==2]=1
    liver3 = np.load('data' + string + '/g_vessel.npy')
    liver3[liver3==3]=1
    liver5 = np.load("data" + string + "/tumor_random1.npy")

    liver3 = skeletonize(liver3)
    liver3[liver3==255]=3
    liver2 = skeletonize(liver2)
    liver2[liver2==255]=2

    matrix = np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))
    matrix += liver1
    matrix += liver2
    matrix[matrix == 3] = 2
    matrix += liver3
    matrix[matrix == 5] = 3
    matrix[matrix == 4] = 3

    correct_flood(matrix, 2, x_l1, y_l1, z_l1)
    correct_flood(matrix, 3, x_l2, y_l2, z_l2)


    if matrix[x_l1][y_l1][z_l1] != 2:
        print('门静脉点不正确')
        sys.exit()
    if matrix[x_l2][y_l2][z_l2] != 3:
        print('肝静脉点不正确')
        sys.exit()




    def reshape(a, b, c):
        a = np.array(a).T
        b = np.array(b).T
        c = np.array(c).T
        m = np.concatenate((a, b, c), axis=1)
        return m


    Linked_liver2=dict()
    Linked_liver3=dict()


    def flood(x, y, z, value, LinkedVessel):
        matrix1 = deepcopy(matrix)
        treetop = []
        def check(x, y, z):
            if matrix1[x, y, z] == value or value + 5:
                return False
            else:
                return True

        q = queue.Queue()
        if check(x, y, z):
            return
        q.put([x, y, z])
        LinkedVessel[(x, y, z)]=[-1,0]
        matrix1[x, y, z] = 6
        x_a = []
        y_b = []
        z_c = []
        for a in range(-1,2,1):
            for b in range(-1,2,1):
                for c in range(-1,2,1):
                    x_a.append(a)
                    y_b.append(b)
                    z_c.append(c)
        count = 0
        while not q.empty():
            count += 1
            p = q.get()
            check1 =[]
            for a in range(-1, 2, 1):
                for b in range(-1, 2, 1):
                    for c in range(-1, 2, 1):
                        check1.append(check(p[0] + a, p[1] + b, p[2] + c))
            flag=1
            for i in range(len(check1)):
                if check1[i]:
                    continue

                x_p = p[0] + x_a[i]
                y_p = p[1] + y_b[i]
                z_p = p[2] + z_c[i]


                if matrix1[x_p, y_p, z_p] == value:
                    flag = 0
                    LinkedVessel[(x_p, y_p, z_p)] = [(p[0], p[1], p[2]), 0]
                    matrix1[x_p, y_p, z_p] = 6
                    q.put([x_p, y_p, z_p])

            if(flag==1):
                treetop.append((p[0],p[1],p[2]))

        l = reshape([np.where(matrix1 == 6)[0]], [np.where(matrix1 == 6)[1]], [np.where(matrix1 == 6)[2]])
        # p = pv.Plotter()
        # p.add_mesh(pv.PolyData(l), color='blue', opacity=1)
        # p.show()
        return treetop

    treetop2=flood(x_l1,y_l1,z_l1,2,Linked_liver2)
    treetop3=flood(x_l2,y_l2,z_l2,3,Linked_liver3)

    vessel_to_point=np.load("data/vessel_to_point.npy",allow_pickle=True).item()
    point_to_vessel=np.load("data/point_to_vessel.npy",allow_pickle=True).item()


    def accumulate(treetop,Linked_liver):
        point_record=dict()
        for p in Linked_liver.keys():
            point_record[p]=np.array([])

        for p in treetop:
            # print(p)
            r=np.array([])
            while(p!=-1):
                if(len(point_record[p])==0 and len(r)!=0):
                    point_record[p]=r
                elif(len(point_record[p])==0 and len(r)==0):
                    pass
                else:
                    point_record[p]=np.vstack((point_record[p],r))

                if(len(r)==0):
                    r=np.array([p])
                else:
                    r=np.vstack((r,p))
                p=Linked_liver[p][0]

        for key,value in point_record.items():
            if(len(value)==0):
                pass
            else:
                value = np.unique(
                    value.view(value.dtype.descr * value.shape[1]),
                )
            # if(key==(x_l1,y_l1,z_l1)):
            #     np.save("test_l2.npy",value)
            #     print(len(value))
            # if(key==(x_l2,y_l2,z_l2)):
            #     print(len(value))
            #     np.save("test_l3.npy",value)

            for p in value:
                Linked_liver[key][1]+=len(vessel_to_point.get((p[0],p[1],p[2]),[]))
            Linked_liver[key][1] += len(vessel_to_point.get(key, []))


    accumulate(treetop2,Linked_liver2)
    accumulate(treetop3,Linked_liver3)

    print(Linked_liver2[(x_l1,y_l1,z_l1)][1])
    print(Linked_liver3[(x_l2,y_l2,z_l2)][1])

    l = reshape([np.where(matrix == 2)[0]], [np.where(matrix == 2)[1]], [np.where(matrix == 2)[2]])
    print(len(l))
    l = reshape([np.where(matrix == 3)[0]], [np.where(matrix == 3)[1]], [np.where(matrix == 3)[2]])
    print(len(l))

    np.save("data/Linked_liver2.npy",Linked_liver2)
    np.save("data/Linked_liver3.npy",Linked_liver3)

