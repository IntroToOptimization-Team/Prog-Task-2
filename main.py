import numpy as np
from simplex import simplex
from interior_point import interior_point

alphas = [0.5, 0.9]
tests = ["test1", "test2", "test3", "test4"]

for test in tests:
    with open(f"tests/{test}.txt") as file:
        # data input
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

        # start test
        print(test)
        # simplex method
        res = simplex(c, A, b, eps)
        print("SIMPLEX METHOD:")
        if res["solver_state"] == "unbounded":
            print("The method is not applicable!")
        else:
            print("The solution:")
            print(" ".join(map(lambda x: str(round(x, 6)), res["x*"])))
            print(round(res["z"], 6))

        print("INTERIOR POINT ALGORITHM:")
        # make input data be in convenient format
        for i in range(len(b)):
            c.append(0)
        identity_matrix = np.eye(len(b))
        A = np.hstack((A, identity_matrix))
        # applying algorithm for different alphas
        for alpha in alphas:
            # interior point algorithm
            result = interior_point(c, A, x, alpha, eps)
            if result is not None:
                z, x = result
                print(f"The approximate optimal value if alpha = {alpha} is {z} at")
                for i in range(len(x)):
                    print(f"x{i + 1} = {x[i]:.4f}")
            else:
                break
        print()





