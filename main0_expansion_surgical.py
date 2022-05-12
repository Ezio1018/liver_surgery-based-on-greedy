from collections import Counter
import numpy as np
from stl import mesh
import math
def expansion_surgical(filepath,size):
    your_mesh = mesh.Mesh.from_file(filepath)
    data = your_mesh.data  # 数据包含平面的法向量和三个点坐标
    vertexs = []
    martix = []

    for p in data:
        vertexs.append(p[1])
    vertexs = np.array(vertexs)  # 保存顶点的坐标
    vertexs_sed = []
    for q in vertexs:
        for p in q:
            vertexs_sed.append(p)
    data = np.array(data)
    count = 0
    c_list1 = np.array([[0, 0, 0]])
    c_list2 = np.array([[0, 0, 0]])
    for p1 in vertexs:
        list1 = []
        for p2 in p1:
            c_l = list(np.argwhere(p2 == c_list1)[:, 0])
            c_d = dict(Counter(c_l))
            indexs = [key for key, value in c_d.items() if value == 3]
            if indexs:
                list1.append(c_list2[indexs[0]])
                continue
            l = list(np.argwhere(p2 == vertexs_sed)[:, 0])
            d = dict(Counter(l))
            list_vertexs = [key for key, value in d.items() if value == 3]
            list_vertexs = np.array(list_vertexs)
            list_vertexs = list_vertexs / 3
            list_vertexs = list_vertexs.astype(int)
            # 存在该点的平面
            # 平面序号
            last_vector = np.array([0.0, 0.0, 0.0])
            for i in list_vertexs:  # 顶点相关的三角面的法向量
                last_vector += data[i][0]  # 顶点的法向量
            grand = math.sqrt(last_vector[0] ** 2 + last_vector[1] ** 2 + last_vector[2] ** 2)  # 归一化底数
            last_vector /= grand
            p3 = p2 + last_vector * size  # 这里决定最后点扩充后的值
            list1.append(p3)
            c_list1 = np.concatenate((c_list1, [p2]))
            c_list2 = np.concatenate((c_list2, [p3]))
            count += 1
        martix.append(list1)
    martix = np.array(martix)
    your_mesh.vectors = martix
    your_mesh.save('data/tumor_enlarged.stl')
