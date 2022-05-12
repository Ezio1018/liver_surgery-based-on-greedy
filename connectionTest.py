import queue
import numpy as np

def correct_flood(matrix, value, x, y, z):

        def check(x, y, z):
            if matrix[x, y, z] == value:
                return False
            else:
                return True

        q = queue.Queue()

        if check(x, y, z):
            print(1)
            print(matrix[x, y, z])
            print("wrong!")
            return
        q.put([x, y, z])
        matrix[x, y, z] = 6
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
            # check1 = [check(p[0] + 1, p[1], p[2]), check(p[0] - 1, p[1], p[2]), check(p[0], p[1] + 1, p[2]),
            #           check(p[0], p[1] - 1, p[2]), check(p[0], p[1], p[2] + 1), check(p[0], p[1], p[2] - 1)]
            check1=[]
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
                if matrix[x_p, y_p, z_p] == value:
                    matrix[x_p, y_p, z_p] = 6
                    q.put([x_p, y_p, z_p])
        matrix[matrix==value]=1
        matrix[matrix==6]=value