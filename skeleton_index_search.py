
import numpy as np
import multiprocessing
from skimage.morphology import skeletonize
from connectionTest import correct_flood


string = ''
liver1 = np.load('data' + string + '/liver.npy')

liver2 = np.load('data' + string + '/m_vessel.npy')
liver2[liver2==2]=1
liver3 = np.load('data' + string + '/g_vessel.npy')
liver3[liver3==3]=1

# liver4 = np.load('file/动脉.npy')
# liver5 = np.load('data/占位' + str(i) + '.npy')
print(liver3.shape)
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


## 生成一组点云的坐标，然后构建点云的mesh

def reshape(a, b, c):
    a = np.array(a).T
    b = np.array(b).T
    c = np.array(c).T
    m = np.concatenate((a, b, c), axis=1)
    return m

def check_sur(p):
    def check(x, y, z):
        if matrix[x, y, z] ==0:
            return False
        else:
            return True
    count = 0
    check1 = [check(p[0] + 1, p[1], p[2]), check(p[0] - 1, p[1], p[2]), check(p[0], p[1] + 1, p[2]),
              check(p[0], p[1] - 1, p[2]), check(p[0], p[1], p[2] + 1), check(p[0], p[1], p[2] - 1)]
    for i in range(len(check1)):
        if check1[i]:
            continue
        count += 1

    return count

l1 = reshape([np.where(matrix == 1)[0]], [np.where(matrix == 1)[1]], [np.where(matrix == 1)[2]])
s_2 = reshape([np.where(liver2 == 2)[0]], [np.where(liver2 == 2)[1]], [np.where(liver2 == 2)[2]])
s_3 = reshape([np.where(liver3 == 3)[0]], [np.where(liver3 == 3)[1]], [np.where(liver3 == 3)[2]])
s2=[]
for p in s_2:
    c=check_sur(p)
    if(c<=2):
        s2.append(p)

s3=[]
for p in s_3:
    c=check_sur(p)
    if(c<=2):
        s3.append(p)
s3 = np.array(s3)


s2 = np.array(s2)
def search_multi(p):
    distance2 = np.sum(np.square(s2 - p), axis=1)
    point2 = s2[np.argmin(distance2)]
    distance3 = np.sum(np.square(s3 - p), axis=1)
    point3 = s3[np.argmin(distance3)]
    p = np.append(p, [point2, point3])
    return p


if __name__ == '__main__':
    pp = multiprocessing.Pool(17)
    b = pp.map(search_multi, l1)
    pp.close()
    pp.join()
    # index=np.array(index)
    np.save('data' + string + '/index_skeleton.npy', b)