import numpy as np
from numpy.linalg import norm


def Inter_Point(C, A, x, alpha, eps):
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

def simplex(C, A, b, eps=0.001):
    # Step 1. Print the optimization problem
    print("PROBLEM:")
    print("max: z = ", end="")
    print(" + ".join([f"{C[i]} * x{i + 1}" for i in range(len(C))]))

    print("subject to the constraints:")
    for row in range(len(b)):
        print(' + '.join([f'{A[row][i]} * x{i + 1}' for i in range(len(A[row]))]).replace("+ -", "- ") + ' <= ' + str(b[row]))

    # Step 2. Initialize
    C = np.concatenate((C, np.zeros(b.shape[0])))

    tableau = np.concatenate((A, np.eye(b.shape[0])), axis=1)
    # current_solution = np.zeros(b.shape[0])
    variable_indexes = list(range(A.shape[1], len(C)))

    # z = np.array([0 for _ in range(len(C))])
    objective_row = -C

    solz = 0

    # Step 3. Iteratively apply the Simplex method
    while True:
        if all(objective_row >= 0):
            # Solution optimal
            solution = np.zeros((tableau.shape[1]))
            for i in range(len(variable_indexes)):
                solution[variable_indexes[i]] = b[i]

            solution = solution[:A.shape[1]]
            return {
                'solver_state': 'solved',
                'x*': list(map(lambda x: round(float(x), 6), solution)),
                # 'z': float(current_solution.T @ b)
                'z': solution.T @ C[:A.shape[1]]
            }

        # Identify the entering variable
        entering_variable_index = np.argmin(objective_row)

        # Identify the leaving variable
        leaving_variable_index = -1
        for index in range(len(b)):
            if tableau[index][entering_variable_index] != 0 and b[index] / tableau[index][entering_variable_index] > 0:
                if leaving_variable_index == -1:
                    leaving_variable_index = index
                else:
                    leaving_variable_index = leaving_variable_index if b[leaving_variable_index] / tableau[
                        leaving_variable_index, entering_variable_index] < b[index] / tableau[
                                                                           index, entering_variable_index] else index

        if leaving_variable_index == -1:
            # problem is unbounded
            return {'solver_state': 'unbounded'}

        # Perform pivot operations
        # leaving_variable_index - row
        # entering_variable_index - column

        variable_indexes[leaving_variable_index] = entering_variable_index
        # current_solution[leaving_variable_index] = C[entering_variable_index]

        b[leaving_variable_index] /= tableau[leaving_variable_index, entering_variable_index]
        tableau[leaving_variable_index] /= tableau[leaving_variable_index, entering_variable_index]

        for row in range(A.shape[0]):
            if row == leaving_variable_index:
                continue

            b[row] -= b[leaving_variable_index] * tableau[row][entering_variable_index]
            tableau[row] -= tableau[leaving_variable_index] * tableau[row][entering_variable_index]

        dsol = objective_row[entering_variable_index] * b[leaving_variable_index]
        objective_row -= tableau[leaving_variable_index] * objective_row[entering_variable_index]
        if -dsol < eps:
            solution = np.zeros((tableau.shape[1]))
            for i in range(len(variable_indexes)):
                solution[variable_indexes[i]] = b[i]

            solution = solution[:A.shape[1]]
            return {
                'solver_state': 'solved',
                'x*': list(map(lambda x: round(float(x), 6), solution)),
                'z': solution.T @ C[:A.shape[1]]
            }



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
            result = Inter_Point(c, A, x, alpha, eps)
            if result is not None:
                z, x = result
                print(f"The approximate optimal value if alpha = {alpha} is {z} at")
                for i in range(len(x)):
                    print(f"x{i + 1} = {x[i]:.4f}")
            else:
                break
        print()





