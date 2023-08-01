from scipy import optimize
import numpy as np
import sympy as sm
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tabulate import tabulate
import ipywidgets as widgets
from sympy import symbols, lambdify
from ipywidgets import interact, interactive, fixed, interact_manual

class SolowModelClass:
    
    def __init__(self, do_print=True):
        """ create the model """

        # if do_print: print('initializing the model:')
        self.par = SimpleNamespace()
        self.val = SimpleNamespace()
        self.sim = SimpleNamespace()

        # if do_print: print('calling .setup()')
        self.setup()
    
    def setup(self):
        """ baseline parameters """

        val = self.val
        par = self.par
        sim = self.sim

        # model parameters for analytical solution
        par.k = sm.symbols('k')
        par.alpha = sm.symbols('alpha')
        par.delta = sm.symbols('delta')
        par.phi =  sm.symbols('phi')
        par.sK = sm.symbols('s_k')
        par.sH = sm.symbols('s_h')
        par.g = sm.symbols('g')
        par.n = sm.symbols('g')
        par.A = sm.symbols('A')
        par.K = sm.symbols('K')
        par.Y = sm.symbols('Y')
        par.L = sm.symbols('L')
        par.k_tilde = sm.symbols('k_tilde')
        par.h_tilde = sm.symbols('h_tilde')
        par.y_tilde = sm.symbols('y_tilde')
        par.k_tilde_ss = sm.symbols('k_tilde_ss')
        par.h_tilde_ss = sm.symbols('h_tilde_ss')

        # model parameter values for numerical solution
        val.sK = 0.1
        val.sH = 0.1
        val.g = 0.05
        val.n = 0.01
        val.alpha = 0.33
        val.delta = 0.3
        val.phi = 0.02

        # simulation parameters for further analysis
        par.simT = 100 #number of periods
        sim.K = np.zeros(par.simT)
        sim.L = np.zeros(par.simT)
        sim.A = np.zeros(par.simT)
        sim.Y = np.zeros(par.simT)
        sim.H = np.zeros(par.simT)
        sim.k_tilde = np.zeros(par.simT)

    def solve_analytical_ss(self):
        """ function that solves the model analytically and returns k_tilde in steady state """

        par = self.par
        # Define transistion equations
        trans_k = sm.Eq(par.k_tilde, 1/((1+par.n)*(1+par.g))*(par.sK*par.k_tilde**par.alpha*par.h_tilde**par.phi+(1-par.delta)*par.k_tilde))
        trans_h = sm.Eq(par.h_tilde, 1/((1+par.n)*(1+par.g))*(par.sH*par.k_tilde**par.alpha*par.h_tilde**par.phi+(1-par.delta)*par.h_tilde))

        # solve the equation for k_tilde and h_tilde
        k_tilde_ss = sm.solve(trans_k, par.k_tilde)[0]
        h_tilde_ss = sm.solve(trans_h, par.h_tilde)[0]

        # Print the solutions
        sm.pprint(k_tilde_ss)
        sm.pprint(h_tilde_ss)

        return k_tilde_ss, h_tilde_ss
    
    def solve_steady_state(self):
        """ function that solves the model numerically and returns k_tilde and h_tilde in steady state """

        par = self.val

        # define the steady state equations
        def steady_state_eq(vars):
            k_tilde, h_tilde = vars
            eq1 = k_tilde - ((1/(1+par.n)*(1+par.g)) * ((par.sK * k_tilde**par.alpha * h_tilde**par.phi + (1-par.delta) * k_tilde)))
            eq2 = h_tilde - ((1/(1+par.n)*(1+par.g)) * ((par.sH * k_tilde**par.alpha * h_tilde**par.phi + (1-par.delta) * h_tilde)))
            return [eq1, eq2]

        # make an initial guess for the solution
        initial_guess = [1, 1]  # initial guess for k_tilde and h_tilde

        # solve the equations numerically
        result = optimize.root(steady_state_eq, initial_guess)
        if result.success:
            k_tilde_ss, h_tilde_ss = result.x
            print("Steady state found:")
            print("k_tilde =", k_tilde_ss)
            print("h_tilde =", h_tilde_ss)

            # Save values
            self.k_tilde_ss = k_tilde_ss
            self.h_tilde_ss = h_tilde_ss

        else:
            # Handle the case when the optimization fails
            k_tilde_ss, h_tilde_ss = None, None
            print("Failed to find steady state.")

        return k_tilde_ss, h_tilde_ss
    
    def convergence(self, evals=10):

        par = self.val
        k, h = 0.5, 0.5
        fks = np.zeros(evals)
        fhs = np.zeros(evals)
        for i in range(evals):
            k_new = ((par.sK*h**par.phi) + (1-par.delta)*k)**(1/(1-par.alpha))
            h_new = ((par.sH*k**par.alpha) + (1-par.g)*h)**(1/par.phi)
            fks[i] = np.abs(k_new - k)
            fhs[i] = np.abs(h_new - h)
            k, h = k_new, h_new
            
        # Plot the convergences for k and h
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        i = evals-1
        axs[0].plot(np.arange(i+1), fks[:i+1], '--', ms=4, color='blue')
        axs[0].set_title("k convergence")
        axs[0].set_xlabel('iteration')
        axs[1].plot(np.arange(i+1), fhs[:i+1], '--', ms=4, color='green')
        axs[1].set_title("h convergence")
        axs[1].set_xlabel('iteration')
        plt.show()
        
        # Return the steady state values of physical and human capital
        return k, h

    def y_steady_state(self):
        par = self.val
        y_ss = self.k_tilde_ss**par.alpha * self.h_tilde_ss**par.phi
        print("Y steady state found=", y_ss)
        return y_ss

    
    