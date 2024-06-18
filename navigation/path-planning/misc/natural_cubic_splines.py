#!/bin/usr/python3

import numpy as np 
import matplotlib.pyplot as plt

def matrices_from_cntrl_pnts(cntrl_pnts):
    #Create the matrix equations that make matrices A and b
    def func(pnt):
        return [pnt**3, pnt**2, pnt, 1]

    b = cntrl_pnts[:, 1]
    A = np.array([func(point) for point in cntrl_pnts[:,0]])

    return (A, b)

def find_coefficients(A, b):
    #Solve Ax = b for x
    return np.linalg.inv(A) @ b  

def cubic_spline(x_coord, coeff):
    #Evaluate y = ax^3 + bx^2 + cx + d 
    return coeff[0]*x_coord**3 + coeff[1]*x_coord**2 + coeff[2]*x_coord + coeff[3]

def plot_spline(coeff, spline_func, scatter=False):
    x_vals = np.linspace(0, 10, 100)
    y_vals = [spline_func(x, coeff) for x in x_vals]

    if scatter:
        plt.scatter(x_vals, y_vals)
    else:
        plt.plot(x_vals, y_vals)
    plt.ylim([0, x_vals[-1]])
    plt.xlim([0, x_vals[-1]])
    plt.show()


if __name__ == "__main__":
    control_points = np.array([[0,1], [2, 1.5], [7, 3], [10, 1]])
    A, b = matrices_from_cntrl_pnts(control_points)

    coefficients = find_coefficients(A, b)
    print(coefficients)

    plot_spline(coefficients, cubic_spline)

