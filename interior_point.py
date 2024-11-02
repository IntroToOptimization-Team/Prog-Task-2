import numpy as np
from numpy.linalg import norm

def interior_point(C, A, x, alpha, eps):
    # i is to stop if we have not enough python accuracy to calculate the answer
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

        # check that x has changed during the step significantly
        if norm(np.subtract(yy, v), ord = 2) < eps:
            return np.dot(C, x), x

    print("The problem does not have solution!")
    return None