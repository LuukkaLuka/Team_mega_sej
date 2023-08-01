import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class RefinedGlobalOptimizer:
    def __init__(self, warm_up_iterations):
        self.warm_up_iterations = warm_up_iterations
        self.bounds = [-600, 600]
        self.tau = 1e-8
        self.max_iterations = 1000
        self.x_opt = np.nan
        self.f_opt = np.inf
        self.x_k0_values = []
        self.iteration_numbers = []

    def griewank(self, x):
        return self.griewank_(x[0], x[1])

    def griewank_(self, x1, x2):
        A = x1**2 / 4000 + x2**2 / 4000
        B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
        return A - B + 1

    def optimize(self):
        for k in range(self.max_iterations):
            x_k = np.random.uniform(self.bounds[0], self.bounds[1], size=2)

            if k >= self.warm_up_iterations:
                chi_k = 0.50 * 2 / (1 + np.exp((k - self.warm_up_iterations) / 100))
                x_k0 = chi_k * x_k + (1 - chi_k) * self.x_opt
            else:
                x_k0 = x_k

            result = optimize.minimize(self.griewank, x_k0, method='BFGS', tol=self.tau)
            x_k_opt = result.x
            f_k_opt = result.fun

            if k == 0 or f_k_opt < self.f_opt:
                self.x_opt = x_k_opt
                self.f_opt = f_k_opt

            self.x_k0_values.append(x_k0)
            self.iteration_numbers.append(k+1)

            if k < 10 or self.f_opt < self.tau:
                print(f'{k:4d}: x^k = ({x_k[0]:7.2f}, {x_k[1]:7.2f}) -> x^k0 = ({x_k0[0]:7.2f}, {x_k0[1]:7.2f})')

            if self.f_opt < self.tau:
                break

    def print_convergence_iteration(self):
        print(f'Converged at iteration: {len(self.iteration_numbers)}')

    def plot_effective_initial_guesses(self):
        x_k0_values = np.array(self.x_k0_values)
        iteration_numbers = np.array(self.iteration_numbers)

        fig, ax = plt.subplots()
        plt.scatter(x_k0_values[:, 0], x_k0_values[:, 1], c=iteration_numbers, cmap='viridis')
        plt.colorbar(label='Iteration Number')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Effective Initial Guesses x^k0 vs. Iteration Number')
        plt.show()
    
    def plot_griewank_surface_contour(self, bound):
        print(f'In [{-bound},{bound}] x [{-bound},{bound}]:')

        # a. grids
        x1_vec = np.linspace(-bound, bound, 1000)
        x2_vec = np.linspace(-bound, bound, 1000)
        x1_grid_griewank, x2_grid_griewank = np.meshgrid(x1_vec, x2_vec, indexing='ij')
        griewank_grid = self.griewank_(x1_grid_griewank, x2_grid_griewank)

        # b. main
        fig = plt.figure(figsize=(12, 4))

        # 3D
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        cs = ax.plot_surface(x1_grid_griewank, x2_grid_griewank, griewank_grid, cmap=cm.jet)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.invert_xaxis()

        # contour
        ax = fig.add_subplot(1, 2, 2)
        cs = ax.contour(x1_vec, x2_vec, griewank_grid, levels=15, cmap=cm.jet)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        plt.show()

