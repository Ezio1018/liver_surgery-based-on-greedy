import numpy as np
import pyvista as pv
import queue
from copy import deepcopy
from collections import Counter
from calculate_surface import calculate_surface
from skimage.morphology import skeletonize

string = ''


def search_dead_cell(matrix, index, value, x, y, z, ii,lll5):
    def reshape(a, b, c):
        a = np.array(a).T
        b = np.array(b).T
        c = np.array(c).T
        m = np.concatenate((a, b, c), axis=1)
        return m

    def flood(x, y, z):
        matrix1 = deepcopy(matrix)

        def check(x, y, z):
            if matrix1[x, y, z] == value or value + 5:
                return False
            else:
                return True

        q = queue.Queue()
        if check(x, y, z):
            return
        q.put([x, y, z])
        matrix1[x, y, z] = 6
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
                    matrix1[x_p, y_p, z_p] = 6
                    q.put([x_p, y_p, z_p])

        l_2 = reshape([np.where(matrix1 == value)[0]], [np.where(matrix1 == value)[1]], [np.where(matrix1 == value)[2]])
        l_7 = reshape([np.where(matrix1 == value + 5)[0]], [np.where(matrix1 == value + 5)[1]],
                      [np.where(matrix1 == value + 5)[2]])
        l_8 = reshape([np.where(matrix1 == value + 6)[0]], [np.where(matrix1 == value + 6)[1]],
                      [np.where(matrix1 == value + 6)[2]])
        l_6 = reshape([np.where(matrix1 == 6)[0]], [np.where(matrix1 == 6)[1]], [np.where(matrix1 == 6)[2]])
        ls = np.concatenate((l_2, l_7), axis=0)
        ls = np.concatenate((l_8, ls), axis=0)
        p = pv.Plotter()
        if(len(ls)!=0 and len(l_6)!=0):
            # p.set_background("white")
            p.add_mesh(pv.PolyData(l_6), color='red', render_points_as_spheres=True, opacity=1)
            p.add_mesh(pv.PolyData(ls), color='blue', render_points_as_spheres=True, opacity=1)
            # p.add_mesh(pv.PolyData(lll5), color='black', render_points_as_spheres=True, opacity=0.2)

            p.show_grid()
            p.show()

        if value == 2:
            np.save('data/死掉的门静脉骨架' + str(ii) + '.npy', ls)
        elif value == 3:
            np.save('data/死掉的肝静脉骨架' + str(ii) + '.npy', ls)
        return ls

    def search(l):
        total_dead = np.array([])

        for p in l:
            # print(p)
            dead = np.array(index.get((p[0], p[1], p[2]), []))
            if (len(dead) == 0):
                continue
            if (len(total_dead) == 0):
                total_dead = dead
            else:
                # print(dead)
                # print(total_dead)
                total_dead = np.concatenate((total_dead, dead))
        return total_dead


    l_d = flood(x, y, z)

    if value == 2:
        dead_cell = search(l_d)
    else:
        dead_cell = search(l_d)

    return dead_cell

# a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
# a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
# l1 = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
# p = pv.Plotter()
# # p.add_mesh(pv.PolyData(l1), color='blue',  render_points_as_spheres=True)
# p.add_mesh(pv.PolyData(l_d), color='yellow', render_points_as_spheres=True)
# p.show()
