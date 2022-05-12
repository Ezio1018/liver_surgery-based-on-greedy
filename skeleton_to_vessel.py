import numpy as np
import multiprocessing
from skimage.morphology import skeletonize
from connectionTest import correct_flood



string = ''
liver1 = np.load('data' + string + '/liver.npy')

liver2 = np.load('data' + string + '/m_vessel.npy')
liver2[liver2 == 2] = 1
liver3 = np.load('data' + string + '/g_vessel.npy')
liver3[liver3 == 3] = 1

# liver4 = np.load('file/动脉.npy')
# liver5 = np.load('data/占位' + str(i) + '.npy')
liver3 = skeletonize(liver3)
liver3[liver3 == 255] = 3
liver2 = skeletonize(liver2)
liver2[liver2 == 255] = 2

matrix = np.zeros((liver1.shape[0], liver1.shape[1], liver1.shape[2]))
matrix += liver1
matrix += liver2
matrix[matrix == 3] = 2
matrix += liver3
matrix[matrix == 5] = 3
matrix[matrix == 4] = 3



def reshape(a, b, c):
    a = np.array(a).T
    b = np.array(b).T
    c = np.array(c).T
    m = np.concatenate((a, b, c), axis=1)
    return m

l2 = reshape([np.where(matrix == 2)[0]], [np.where(matrix == 2)[1]], [np.where(matrix == 2)[2]])
l3 = reshape([np.where(matrix == 3)[0]], [np.where(matrix == 3)[1]], [np.where(matrix == 3)[2]])
liver2 = np.load('data' + string + '/m_vessel.npy')
l2_vessel = reshape([np.where(liver2 == 2)[0]], [np.where(liver2 == 2)[1]], [np.where(liver2 == 2)[2]])
liver3 = np.load('data' + string + '/g_vessel.npy')
l3_vessel = reshape([np.where(liver3 == 3)[0]], [np.where(liver3 == 3)[1]], [np.where(liver3 == 3)[2]])



def search_multi2(p,):
    distance2 = np.sum(np.square(l2 - p), axis=1)
    point2 = l2[np.argmin(distance2)]
    p = np.append(p, [point2])
    return p

def search_multi3(p,):
    distance3 = np.sum(np.square(l3 - p), axis=1)
    point3 = l3[np.argmin(distance3)]
    p = np.append(p, [point3])
    return p


if __name__ == '__main__':
    pp = multiprocessing.Pool(17)
    b = pp.map(search_multi2, l2_vessel)
    c = pp.map(search_multi3, l3_vessel)

    pp.close()
    pp.join()

    np.save('data' + string + '/skeleton_to_vessel2.npy', b)
    np.save('data' + string + '/skeleton_to_vessel3.npy', c)
