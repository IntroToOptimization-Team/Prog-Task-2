import numpy as np
from numpy.linalg import norm


def Inter_Point(C, A, x, alpha, eps):
    i = 1
    if min(x) < 0 or not (0 < alpha < 1) or eps <= 0:
        print("The method is not applicable!")
        return None
    while i < 1000:
        v = x
        D = np.diag(x)

        AA = np.dot(A, D)
        cc = np.dot(D, C)

        I = np.eye(len(C))

        F = np.dot(AA, np.transpose(AA))
        FI = np.linalg.inv(F)

        H = np.dot(np.transpose(AA), FI)

        P = np.subtract(I, np.dot(H, AA))

        cp = np.dot(P, cc)

        nu = np.absolute(np.min(cp))
        y = np.add(np.ones(len(x), float), (alpha / nu) * cp)
        yy = np.dot(D, y)

        x = yy

        i += 1

        if norm(np.subtract(yy, v), ord = 2) < eps:
            return np.dot(C, x), x

    print("The problem does not have solution!")
    return None


alphas = [0.5, 0.9]
tests = ["test1", "test2", "test3", "test4"]

for test in tests:
    with open(f"tests/{test}.txt") as file:
        c = [float(el) for el in file.readline().strip().split(" ")]
        empty_line = file.readline()
        A = []
        temp = file.readline().strip()
        while temp != "":
            A.append([float(el) for el in temp.strip().split(" ")])
            temp = file.readline().strip()
        A = np.array(A)
        x = np.array([float(el) for el in file.readline().strip().split(" ")])
        empty_line = file.readline()
        b = np.array([float(el) for el in file.readline().strip().split(" ")])
        empty_line = file.readline()
        eps = float(file.readline().strip())

        print(test)

        for alpha in alphas:
            result = Inter_Point(c, A, x, alpha, eps)
            if result is not None:
                z, x = result
                print(f"The approximate optimal value if alpha = {alpha} is {z} at")
                for i in range(len(x)):
                    print(f"x{i + 1} = {x[i]:.4f}")
            else:
                break

