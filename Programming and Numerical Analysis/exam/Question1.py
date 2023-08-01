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

class OptimalLCalculator:
    def __init__(self):
        # Defining the parameter symbols
        self.L, self.w_tilde, self.kappa, self.alpha, self.nu, self.C, self.G, self.tau = sm.symbols('L w_tilde kappa alpha nu C G tau')

        # Defining utility & consumption constraint (we have already substituted the expression for w_tilde in the consumption constraint)
        self.utility = sm.log(self.C**self.alpha * self.G**(1-self.alpha)) - (self.nu * (self.L**2 / 2))
        self.consumption_constraint = sm.Eq(self.C, self.kappa + self.w_tilde * self.L)

    def calculate_optimal_L(self, G_values):
        optimal_L_values = []
        for G_val in G_values:
            # Solve the consumption constraint for C
            consumption_constraint_solved = sm.solve(self.consumption_constraint.subs(self.G, G_val), self.C)

            # Substitute the expression for C in the utility function
            utility_subs = self.utility.subs(self.C, consumption_constraint_solved[0])

            # Solve for optimal L
            optimal_L = sm.solve(sm.Eq(sm.diff(utility_subs.subs(self.G, G_val), self.L), 0), self.L)[0]
            optimal_L_values.append(optimal_L)

        return optimal_L_values
    
    def plot_L_star(self, kappa, alpha, nu, tau, w):
        w_tilde = (1 - tau) * w
        L_star = (-kappa + np.sqrt(kappa**2 + 4 * alpha / nu * w_tilde**2)) / (2 * w_tilde)

        plt.plot(w, L_star)
        plt.xlabel('w_tilde')
        plt.ylabel('L^star')
        plt.title('Plot of L^star(w_tilde)')
        plt.xlim(0, max(w))
        plt.ylim(0, max(L_star))
        plt.grid(True)
        plt.show()
    
    def plot_question3(self, kappa, alpha, nu, w):
        def L_star(kappa, alpha, nu, w_tilde):
            return (-kappa + np.sqrt(kappa**2 + 4 * alpha / nu * w_tilde**2)) / (2 * w_tilde)

        def G(kappa, alpha, nu, tau, w):
            w_tilde = (1 - tau) * w
            return tau * w * L_star(kappa, alpha, nu, w_tilde)*((1- tau)*w)

        def L_starG(kappa, alpha, nu, tau, w):
            return G(kappa, alpha, nu, tau, w) / (tau*(1-tau)*w**2)

        def V(kappa, alpha, nu, tau, w):
            L = np.linspace(0, 24, 100)  # Grid of L values
            C = kappa + (1 - tau) * w * L
            utility = np.log(C**alpha * G(kappa, alpha, nu, tau, w)**(1 - alpha)) - nu * L**2 / 2
            return np.max(utility)

        tau_values = np.linspace(0, 1, 100)  # Grid of tau values

        L_values = L_starG(kappa, alpha, nu, tau_values, w)
        G_values = G(kappa, alpha, nu, tau_values, w)
        V_values = np.zeros_like(tau_values)

        for i, tau in enumerate(tau_values):
            V_values[i] = V(kappa, alpha, nu, tau, w)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        plt.plot(tau_values, L_values)
        plt.xlabel('tau')
        plt.ylabel('L^star')
        plt.title('Plot of L^star')

        plt.subplot(2, 2, 2)
        plt.plot(tau_values, G_values)
        plt.xlabel('tau')
        plt.ylabel('G')
        plt.title('Plot of G')

        plt.subplot(2, 2, 3)
        plt.plot(tau_values, V_values)
        plt.xlabel('tau')
        plt.ylabel('V')
        plt.title('Plot of V')

        plt.tight_layout()
        plt.show()

    def plot_question4(self, kappa, alpha, nu, w):
        def V(tau, w, kappa, alpha, nu):
            w_tilde = (1 - tau) * w
            L_star = (-kappa + np.sqrt(kappa**2 + 4 * alpha/nu * w_tilde**2)) / (2 * w_tilde)
            C = kappa + (1 - tau) * w * L_star
            G = tau * w * L_star*((1-tau)*w)
            return np.log(C**alpha * G**(1 - alpha)) - nu * L_star**2 / 2
        
        def maximize_V(w, kappa, alpha, nu):
            result = minimize_scalar(lambda tau: -V(tau, w, kappa, alpha, nu), bounds=(0, 1), method='bounded')
            return result.x

        max_tau = maximize_V(w, kappa, alpha, nu)
        max_V = V(max_tau, w, kappa, alpha, nu)

        tau_vals = np.linspace(0, 1, 100)
        V_vals = [V(tau, w, kappa, alpha, nu) for tau in tau_vals]

        # Plot V(tau)
        plt.plot(tau_vals, V_vals)
        plt.scatter(max_tau, max_V, color='red', label='Max V')
        plt.xlabel('tau')
        plt.ylabel('V')
        plt.legend()
        plt.title('Objective Function V(tau)')
        plt.show()

        return max_tau, max_V




