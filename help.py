import numpy as np
from numpy.linalg import norm

x = np.array([2, 2, 4, 3], float)
alpha = 0.5
A = np.array([[2, -2, 8, 0], [-6, -1, 0, -1]], float)
c = np.array([-2, 3, 0, 0], float)

i = 0
eps = 0.00001
while i < 1000:
    print(x)
    v = x
    D = np.diag(x)

    AA = np.dot(A, D)
    cc = np.dot(D, c)

    I = np.eye(4)

    F = np.dot(AA, np.transpose(AA))
    FI = np.linalg.inv(F)

    H = np.dot(np.transpose(AA), FI)

    P = np.subtract(I, np.dot(H, AA))

    cp = np.dot(P, cc)

    nu = np.absolute(np.min(cp))
    y = np.add(np.ones(4, float), (alpha / nu) * cp)
    yy = np.dot(D, y)

    x = yy

    i += 1

    if norm(np.subtract(yy, v), ord=2) < eps:
        print(np.dot(c, x), x)