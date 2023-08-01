import math 
import scipy.optimize
import numpy as np
import pandas as pd
import sympy as sm
from scipy import optimize
from tabulate import tabulate
from scipy import optimize
import scipy.optimize as optimize
from sympy import symbols, lambdify
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt

class hairdresser:
    def __init__(self, eta, w, kappa_values):
        self.eta = eta
        self.w = w
        self.kappa_values = kappa_values
    
    def calculate_optimal_ell(self):
        p_t = sm.symbols('p_t')
        y_t = sm.symbols('y_t')
        w = sm.symbols('w')
        kappa_t = sm.symbols('kappa_t')
        ell_t = sm.symbols('ell_t')
        eta = sm.symbols('eta')

        # Define the profit function
        profit_function = kappa_t * ell_t**(1 - eta) - w * ell_t

        # Calculate the derivative of the profit function with respect to ell
        derivative = sm.diff(profit_function, ell_t)

        # Set the derivative equal to zero and solve for ell
        optimal_ell = sm.solve(derivative, ell_t)

        return optimal_ell

    def calculate_expected_value(self):
        eta = 0.5
        w = 1.0
        rho = 0.90
        iota = 0.01
        sigma_epsilon = 0.10
        R = (1 + 0.01) ** (1 / 12)
        K = 1000  # Number of shock series

        # Simulate shock series
        np.random.seed(1)
        epsilon_series = np.random.normal(-0.5 * sigma_epsilon**2, sigma_epsilon, size=(K, 120))

        # Calculate value function for each shock series
        value_functions = []
        for k in range(K):
            kappa_series = np.exp(np.zeros(120))
            ell_series = np.zeros(120)
            value_function = 0

            for t in range(120):
                kappa_series[t] = np.exp(rho * np.log(kappa_series[t - 1]) + epsilon_series[k, t])
                ell_series[t] = (((1 - eta) * kappa_series[t]) / w ) ** (1 / eta)

                if t > 0 and ell_series[t] != ell_series[t - 1]:
                    value_function += R ** (-t) * (kappa_series[t] * ell_series[t] ** (1 - eta) - w * ell_series[t]- iota)

            value_functions.append(value_function)

        # Calculate expected value of the salon
        H = np.mean(value_functions[:-1])
        return H

    def calculate_new_expected_value(self):
        eta = self.eta
        w = self.w
        rho = 0.90
        iota = 0.01
        sigma_epsilon = 0.10
        R = (1 + 0.01) ** (1 / 12)
        K = 1000  # Number of shock series
        delta = 0.05  # Adjustment threshold

        # Simulate shock series
        np.random.seed(1)
        epsilon_series = np.random.normal(-0.5 * sigma_epsilon**2, sigma_epsilon, size=(K, 120))

        # Calculate value function for each shock series
        value_functions = []
        for k in range(K):
            kappa_series = np.exp(np.zeros(120))
            ell_series = np.zeros(120)
            value_function = 0

            for t in range(120):
                kappa_series[t] = np.exp(rho * np.log(kappa_series[t - 1]) + epsilon_series[k, t])

                ell_star = (((1 - eta) * kappa_series[t]) / w )** (1 / eta)

                if t == 0 or abs(ell_series[t - 1] - ell_star) > delta:
                    ell_series[t] = ell_star
                else:
                    ell_series[t] = ell_series[t - 1]

                if t > 0 and ell_series[t] != ell_series[t - 1]:
                    value_function += R ** (-t) * (kappa_series[t] * ell_series[t] ** (1 - eta) - w * ell_series[t] - iota)

            value_functions.append(value_function)

        # Calculate expected value of the salon
        H = np.mean(value_functions[:-1])
        return H
    
    def optimize_delta(self):
        eta = self.eta
        w = self.w
        rho = 0.90
        iota = 0.01
        sigma_epsilon = 0.10
        R = (1 + 0.01) ** (1 / 12)
        K = 1000  # Number of shock series

        delta_values = np.linspace(0, 1, 100)  # Range of delta values to search over
        H_values = []

        # Simulate shock series
        np.random.seed(1)
        epsilon_series = np.random.normal(-0.5 * sigma_epsilon ** 2, sigma_epsilon, size=(K, 120))

        # Calculate value function for each delta value
        for delta in delta_values:
            value_functions = []
            for k in range(K):
                kappa_series = np.exp(np.zeros(120))
                ell_series = np.zeros(120)
                value_function = 0

                for t in range(120):
                    kappa_series[t] = np.exp(rho * np.log(kappa_series[t - 1]) + epsilon_series[k, t])

                    ell_star = (((1 - eta) * kappa_series[t]) / w ) ** (1 / eta)

                    if t == 0 or abs(ell_series[t - 1] - ell_star) > delta:
                        ell_series[t] = ell_star
                    else:
                        ell_series[t] = ell_series[t - 1]

                    if t > 0 and ell_series[t] != ell_series[t - 1]:
                        value_function += R ** (-t) * (
                                kappa_series[t] * ell_series[t] ** (1 - eta) - w * ell_series[t] - iota)

                value_functions.append(value_function)

            # Calculate expected value of the salon for the current delta value
            H = np.mean(value_functions)
            H_values.append(H)

        # Find the optimal delta that maximizes H
        optimal_delta = delta_values[np.argmax(H_values)]
        max_H = np.max(H_values)

        # Plot the results
        plt.plot(delta_values, H_values)
        plt.xlabel('Delta')
        plt.ylabel('Expected value of the salon (H)')
        plt.title('Optimization of Delta')
        plt.axvline(x=optimal_delta, color='r', linestyle='--', label='Optimal Delta')
        plt.xlim(-0.01, 1)
        plt.legend()
        plt.show()

        print(f"Optimal Delta: {optimal_delta}")
        print(f"Maximum H value: {max_H}")

    def calculate_new_expected_value_with_lag(self, lag_weight):
        eta = self.eta
        w = self.w
        rho = 0.90
        iota = 0.01
        sigma_epsilon = 0.10
        R = (1 + 0.01) ** (1 / 12)
        K = 1000  # Number of shock series

        # Simulate shock series
        np.random.seed(1)
        epsilon_series = np.random.normal(-0.5 * sigma_epsilon ** 2, sigma_epsilon, size=(K, 120))

        # Calculate value function for each shock series
        value_functions = []
        for k in range(K):
            kappa_series = np.exp(np.zeros(120))
            ell_series = np.zeros(120)
            value_function = 0

            for t in range(120):
                kappa_series[t] = np.exp(rho * np.log(kappa_series[t - 1]) + epsilon_series[k, t])

                ell_star = (((1 - eta) * kappa_series[t]) / w ) ** (1 / eta)

                if t == 0:
                    ell_series[t] = ell_star
                else:
                    ell_series[t] = lag_weight * ell_star + (1 - lag_weight) * ell_series[t - 1]

                if t > 0 and ell_series[t] != ell_series[t - 1]:
                    value_function += R ** (-t) * (
                            kappa_series[t] * ell_series[t] ** (1 - eta) - w * ell_series[t] - iota)

            value_functions.append(value_function)

        # Calculate expected value of the salon
        H = np.mean(value_functions[:-1])
        return H

