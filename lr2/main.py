import numpy as np


class QuadraticProblem:
    def __init__(self, c, D, A, b):
        self.c = np.array(c, dtype=float)
        self.D = np.array(D, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.n = len(c)
        self.m = len(b)

        self._validate()

    def _validate(self):
        if not np.allclose(self.D, self.D.T):
            raise ValueError("Матрица D должна быть симметричной!")

        eigenvalues = np.linalg.eigvalsh(self.D)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                f"Матрица D должна быть положительно полуопределённой! "
                f"Собственные значения: {eigenvalues}"
            )

    def prepare_linear_system(self):
        row1_x = 2 * self.D
        row1_l = self.A.T
        row1_v = -np.eye(self.n)
        row1_w = np.zeros((self.n, self.m))
        rhs1 = -self.c
        row2_x = self.A
        row2_l = np.zeros((self.m, self.m))
        row2_v = np.zeros((self.m, self.n))
        row2_w = np.eye(self.m)
        rhs2 = self.b
        M_top = np.hstack([row1_x, row1_l, row1_v, row1_w])
        M_bot = np.hstack([row2_x, row2_l, row2_v, row2_w])
        M = np.vstack([M_top, M_bot])
        B = np.concatenate([rhs1, rhs2])

        return M, B

    def get_complementary_pairs(self):
        pairs = []
        for j in range(self.n):
            idx_x = j
            idx_v = self.n + self.m + j
            pairs.append((idx_x, idx_v))
        for i in range(self.m):
            idx_lambda = self.n + i
            idx_w = self.n + self.m + self.n + i
            pairs.append((idx_lambda, idx_w))
        return pairs

    def build_auxiliary_lp(self, M, B):
        num_eq = M.shape[0]
        num_vars_original = M.shape[1]
        M_adj = M.copy()
        B_adj = B.copy()
        for i in range(num_eq):
            if B_adj[i] < 0:
                M_adj[i, :] = -M_adj[i, :]
                B_adj[i] = -B_adj[i]

        I_artificial = np.eye(num_eq)
        M_aux = np.hstack([M_adj, I_artificial])
        B_aux = B_adj

        c_aux = np.zeros(num_vars_original + num_eq)
        c_aux[num_vars_original:] = 1

        return M_aux, B_aux, c_aux

    def objective_value(self, x):
        return self.c @ x + x @ self.D @ x


class ModifiedSimplex:
    def __init__(self, M, B, c, complementary_pairs, max_iterations=100):
        self.M = M.copy()
        self.B = B.copy()
        self.c = c.copy()
        self.pairs = complementary_pairs
        self.max_iterations = max_iterations

        self.num_eq = M.shape[0]
        self.num_vars = M.shape[1]

        self.basis = list(range(self.num_vars - self.num_eq, self.num_vars))

        self.tableau = None
        self.iterations = 0

    def get_tableau(self):
        B_mat = self.M[:, self.basis]
        B_inv = np.linalg.inv(B_mat)
        c_B = self.c[self.basis]

        tableau = np.zeros((self.num_eq + 1, self.num_vars + 1))
        tableau[:self.num_eq, :self.num_vars] = B_inv @ self.M
        tableau[:self.num_eq, -1] = B_inv @ self.B
        tableau[-1, :self.num_vars] = c_B @ B_inv @ self.M - self.c
        tableau[-1, -1] = c_B @ B_inv @ self.B

        return tableau

    def get_basis_values(self):
        values = np.zeros(self.num_vars)
        B_mat = self.M[:, self.basis]
        B_inv = np.linalg.inv(B_mat)
        x_B = B_inv @ self.B
        for i, idx in enumerate(self.basis):
            values[idx] = x_B[i]
        return values

    def is_complementary_violated(self, entering_idx):
        for pair in self.pairs:
            if entering_idx in pair:
                partner_idx = pair[0] if pair[1] == entering_idx else pair[1]
                if partner_idx in self.basis:
                    partner_pos = self.basis.index(partner_idx)
                    B_mat = self.M[:, self.basis]
                    B_inv = np.linalg.inv(B_mat)
                    x_B = B_inv @ self.B
                    if x_B[partner_pos] > 1e-10:
                        return True
        return False

    def find_entering_variable(self, tableau):
        for j in range(self.num_vars):
            if tableau[-1, j] > 1e-10:
                if not self.is_complementary_violated(j):
                    return j
        return None

    def find_leaving_variable(self, tableau, entering_idx):
        min_ratio = float('inf')
        leaving_pos = -1
        for i in range(self.num_eq):
            if tableau[i, entering_idx] > 1e-10:
                ratio = tableau[i, -1] / tableau[i, entering_idx]
                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving_pos = i
        return leaving_pos

    def solve(self, verbose=True):
        for iteration in range(self.max_iterations):
            self.iterations = iteration + 1
            tableau = self.get_tableau()
            self.tableau = tableau

            if verbose:
                print(f"\n--- Итерация {iteration + 1} ---")
                print(f"Базис: {self.basis}")
                print(f"Значение F(z): {tableau[-1, -1]:.6f}")

            if np.all(tableau[-1, :-1] <= 1e-10):
                if verbose:
                    print("\nДостигнута оптимальность")
                break

            entering_idx = self.find_entering_variable(tableau)
            if entering_idx is None:
                if verbose:
                    print("\nНет допустимой входящей переменной "
                          "условия дополняющей нежёсткости не позволяют продолжить")
                break
            if verbose:
                print(f"Входящая переменная: x_{entering_idx}")
            leaving_pos = self.find_leaving_variable(tableau, entering_idx)
            if leaving_pos == -1:
                if verbose:
                    print("\nЗадача неограничена!")
                return None
            leaving_idx = self.basis[leaving_pos]
            if verbose:
                print(f"Исходящая переменная: x_{leaving_idx} "
                      f"(позиция {leaving_pos} в базисе)")
            self.basis[leaving_pos] = entering_idx
        values = self.get_basis_values()
        objective_value = self.tableau[-1, -1] if self.tableau is not None else float('inf')
        if verbose:
            print(f"F(z) = {objective_value:.6f}")
        return values, objective_value

def solve_quadratic_program(c, D, A, b, verbose=True):
    try:
        qp = QuadraticProblem(c, D, A, b)
    except ValueError as e:
        print(f"\nОшибка входных данных: {e}")
        return None

    M, B = qp.prepare_linear_system()
    if verbose:
        print(f"\nСистема Куна–Таккера построена")
        print(f" Размер матрицы M: {M.shape}")

    pairs = qp.get_complementary_pairs()
    if verbose:
        print(f"\nПары дополняющей нежёсткости: {pairs}")

    M_aux, B_aux, c_aux = qp.build_auxiliary_lp(M, B)
    if verbose:
        print(f"\nВспомогательная задача ЛПП построена")
        print(f"  Размер расширенной матрицы: {M_aux.shape}")

    simplex = ModifiedSimplex(M_aux, B_aux, c_aux, pairs, max_iterations=100)
    result = simplex.solve(verbose=verbose)

    if result is None:
        print("\n✗ Не удалось найти решение!")
        return None

    solution, objective_value = result

    if objective_value > 1e-6:
        print(f"\nЗадача КПП не имеет решения (F(z) = {objective_value:.6f} > 0)")
        return None

    print(f"\nЗадача КПП решена! (F(z) = {objective_value:.6f} ≈ 0)")

    x_solution = solution[:qp.n]

    if verbose:
        print(f"\nОптимальное решение x* = {x_solution}")
        f_value = qp.objective_value(x_solution)
        print(f"Значение целевой функции f(x*) = {f_value:.6f}")

    return x_solution


if __name__ == "__main__":
    print("Пример")
    print("f(x) = 2x1^2 + 2x1*x2 + 2x2^2 - 4x1 - 6x2 -> min")
    print("x1 + 2x2 <= 2,  x1 >= 0,  x2 >= 0")
    c = [-4, -6]
    D = [[2, 1],
         [1, 2]]
    A = [[1, 2]]
    b = [2]
    solution = solve_quadratic_program(c, D, A, b, verbose=True)
    if solution is not None:
        print("ответ:")
        print(f"  x₁* = {solution[0]:.6f}  (ожидается: 1/3 ≈ 0.333333)")
        print(f"  x₂* = {solution[1]:.6f}  (ожидается: 5/6 ≈ 0.833333)")
        print(f"  f*  = {QuadraticProblem(c, D, A, b).objective_value(solution):.6f}"
              f"  (ожидается: -4 1/6 ≈ -4.166667)")

    print("задача 1")
    print("f(x) = 2x1^2 + 3x2^2 + 4x1*x2 - 6x1 - 3x2 -> min")
    print("x1 + x2 <= 1, 2x1 + 3x2 <= 4, x1 >= 0,  x2 >= 0")
    c = [-6, -3]
    D = [[2, 2],
         [2, 3]]
    A = [[1, 1],[2, 3]]
    b = [1,4]
    solution = solve_quadratic_program(c, D, A, b, verbose=True)
    if solution is not None:
        print("ответ:")
        print(f"  x₁* = {solution[0]:.6f}")
        print(f"  x₂* = {solution[1]:.6f}")
        print(f"  f*  = {QuadraticProblem(c, D, A, b).objective_value(solution):.6f}")

    print("задача 2")
    print("-f(x) = -x1 - 2x2 + x2^2 -> min")
    print("3x1 + 2x2 <= 6, x1 + 2x2 <= 4, x1 >= 0,  x2 >= 0")
    c = [-1, -2]
    D = [[0, 0],
         [0, 1]]
    A = [[3, 2],[1, 2]]
    b = [6,4]
    solution = solve_quadratic_program(c, D, A, b, verbose=True)
    if solution is not None:
        print("ответ:")
        print(f"  x₁* = {solution[0]:.6f}")
        print(f"  x₂* = {solution[1]:.6f}")
        print(f"  f*  = {QuadraticProblem(c, D, A, b).objective_value(solution):.6f}")