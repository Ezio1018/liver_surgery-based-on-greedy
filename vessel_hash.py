import numpy as np
from collections import Counter

def hash_vessel():
    index = np.load("data/index_skeleton.npy")

    m1 = index[:, 0:3]
    m2 = index[:, 3:6]
    m3 = index[:, 6:9]

    vessel_to_point = dict()

    for point in m2:
        if ((point[0], point[1], point[2]) in vessel_to_point.keys()):
            pass
        else:
            l = list(np.argwhere(m2 == [point[0], point[1], point[2]])[:, 0])
            d = dict(Counter(l))
            p = [key for key, value in d.items() if value == 3]
            a = [m1[i] for i in p]
            vessel_to_point[(point[0], point[1], point[2])] = a

    for point in m3:
        if ((point[0], point[1], point[2]) in vessel_to_point.keys()):
            pass
        else:
            l = list(np.argwhere(m3 == [point[0], point[1], point[2]])[:, 0])
            d = dict(Counter(l))
            p = [key for key, value in d.items() if value == 3]
            a = [m1[i] for i in p]
            vessel_to_point[(point[0], point[1], point[2])] = a

    point_to_vessel = dict()

    for i in range(len(m1)):
        point = m1[i]
        point_to_vessel[(point[0], point[1], point[2])] = [m2[i], m3[i]]

    np.save("data/vessel_to_point.npy", vessel_to_point)
    np.save("data/point_to_vessel.npy", point_to_vessel)

    skeleton_to_vessel2 = np.load("data/skeleton_to_vessel2.npy")
    skeleton_to_vessel2_dict=dict()

    m1 = skeleton_to_vessel2[:, 0:3]
    m2 = skeleton_to_vessel2[:, 3:6]

    for point in m2:
        if ((point[0], point[1], point[2]) in skeleton_to_vessel2_dict.keys()):
            pass
        else:
            l = list(np.argwhere(m2 == [point[0], point[1], point[2]])[:, 0])
            d = dict(Counter(l))
            p = [key for key, value in d.items() if value == 3]
            a = [m1[i] for i in p]
            skeleton_to_vessel2_dict[(point[0], point[1], point[2])] = a

    skeleton_to_vessel3 = np.load("data/skeleton_to_vessel3.npy")
    skeleton_to_vessel3_dict=dict()

    m1 = skeleton_to_vessel3[:, 0:3]
    m2 = skeleton_to_vessel3[:, 3:6]

    for point in m2:
        if ((point[0], point[1], point[2]) in skeleton_to_vessel3_dict.keys()):
            pass
        else:
            l = list(np.argwhere(m2 == [point[0], point[1], point[2]])[:, 0])
            d = dict(Counter(l))
            p = [key for key, value in d.items() if value == 3]
            a = [m1[i] for i in p]
            skeleton_to_vessel3_dict[(point[0], point[1], point[2])] = a

    np.save("data/skeleton_to_vessel2_dict.npy", skeleton_to_vessel2_dict)
    np.save("data/skeleton_to_vessel3_dict.npy", skeleton_to_vessel3_dict)
