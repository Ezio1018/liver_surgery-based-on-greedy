import numpy as np
from skimage.morphology import skeletonize
import pyvista as pv
import matplotlib.pyplot as plt

liver2 = np.load(r"data/m_vessel.npy")
liver3 = np.load(r'data/g_vessel.npy')
liver3[liver3==3]=1
liver2[liver2==2]=1
liver3 = skeletonize(liver3)
liver3[liver3 == 255] = 3
liver2 = skeletonize(liver2)
liver2[liver2 == 255] = 2

def reshape(a, b, c):
    a = np.array(a).T
    b = np.array(b).T
    c = np.array(c).T
    m = np.concatenate((a, b, c), axis=1)
    return m

l2 = reshape([np.where(liver2 == 2)[0]], [np.where(liver2 == 2)[1]], [np.where(liver2 == 2)[2]])
l3 = reshape([np.where(liver3 == 3)[0]], [np.where(liver3 == 3)[1]], [np.where(liver3 == 3)[2]])
p = pv.Plotter()

def show_cor(cor):
    print(cor)
p.set_background("white")
# p.add_mesh(pv.PolyData(l2), color='red', opacity=1)
p.add_mesh(pv.PolyData(l3), color='blue', opacity=1)
# p.add_mesh(pv.PolyData(c), color='green', opacity=1)

p.enable_point_picking(callback=show_cor)
p.show_grid()
p.show()

reader = pv.get_reader("data/g_vessel.stl")
mesh = reader.read()


mesh.plot(color='blue',background="white")