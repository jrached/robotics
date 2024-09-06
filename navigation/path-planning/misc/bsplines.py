#!/bin/usr/python3

"""
Functions for creating B-Splines in 2D and 3D given the control points.
TODO: 1. Modify make_bsplines_3d() function so that one function can make both the 2D and 3D splines.
      2. Make this into a class.
      3. Create a from piecewise polynomial to control points method. 
"""

import numpy as np
import matplotlib.pyplot as plt

#####################################
# Making the B-Splines
#####################################
def compute_basis(i, k, t, knots, memo):
    """
    Computes the value of the ith basis of order k at point t.

    inputs: 
        - i: the index of the basis.
        - k: the order of the basis.
        - t: the point the basis is evaluated in.
        - knots: the knot vector of the spline.
        - memo: the lookup table for dp alg.

    output: 
        - basis: the value of the ith basis of order k at point t.
    """
    #base case 
    if k == 1:
        if t >= knots[i] and t < knots[i+1]: return 1
        else: return 0

    #check memo
    if (i, k) in memo: 
        return memo[(i,k)]
    
    #recursive step
    a = (t - knots[i]) / (knots[i + k - 1] - knots[i])
    b = (knots[i+k] - t) / (knots[i+k] - knots[i+1])

    N_1 = compute_basis(i, k-1, t, knots, memo)
    N_2 = compute_basis(i+1, k-1, t, knots, memo)

    memo[(i, k)] = a * N_1 + b * N_2

    return memo[(i, k)] 

def compute_bases(k, t, num_ctrl_pnts, knots):
    """
    Computes the values of the n + 1 bases of order k at point t.

    inputs: 
        - k: order of spline.
        - t: value at which the function is being evaluated.
        - num_ctrl_pnts: n + 1, the number of control points for the spline.
        - knots: the knot vector of the spline.

    output: 
        - bases: a list of size n + 1 made up of the values of the bases of order k at point t.
    """
    bases, memo = [], {}
    for i in range(num_ctrl_pnts):
        basis = compute_basis(i, k, t, knots, memo)
        bases.append(basis)

    return bases


def make_bspline_2d(t_vals, ctrl_pnts, k, knots):
    """
    Computes a bspline given the control points and the order of the bspline.

    inputs: 
        - t_vals: values for which to compute the spline.
        - ctrl_pnts: list of (x, y) tuples.
        - k: order of bspline. k = p + 1, where p is the degree of the polynomial.
        - knots: the knot vector of the bspline. V = (t0, t1, ..., t_{n+k})
   
    output:
        - P_ts: the x and y value of the bspline at each knot vector. This is a
                list of (P_x(t_i), P_y(t_i)) tuples. 
    """

    num_ctrl_pnts = len(ctrl_pnts)

    assert num_ctrl_pnts >= k, "Number of control points must be at least k"

    P_ts = []
    for t in t_vals:
        bases = compute_bases(k, t, num_ctrl_pnts, knots)
        
        P_xt = sum([basis * ctrl_pnt[0] for basis, ctrl_pnt in zip(bases, ctrl_pnts)])
        P_yt = sum([basis * ctrl_pnt[1] for basis, ctrl_pnt in zip(bases, ctrl_pnts)])
        P_ts.append((P_xt, P_yt))
        
    return P_ts

def make_bspline_3d(t_vals, ctrl_pnts, k, knots):
    """
    Computes a bspline given the control points and the order of the bspline.

    inputs: 
        - t_vals: values for which to compute the spline.
        - ctrl_pnts: list of (x, y) tuples.
        - k: order of bspline. k = p + 1, where p is the degree of the polynomial.
        - knots: the knot vector of the bspline. V = (t0, t1, ..., t_{n+k})
   
    output:
        - P_ts: the x and y value of the bspline at each knot vector. This is a
                list of (P_x(t_i), P_y(t_i)) tuples. 
    """

    num_ctrl_pnts = len(ctrl_pnts)

    assert num_ctrl_pnts >= k, "Number of control points must be at least k"

    P_ts = []
    for t in t_vals:
        bases = compute_bases(k, t, num_ctrl_pnts, knots)
        
        P_xt = sum([basis * ctrl_pnt[0] for basis, ctrl_pnt in zip(bases, ctrl_pnts)])
        P_yt = sum([basis * ctrl_pnt[1] for basis, ctrl_pnt in zip(bases, ctrl_pnts)])
        P_zt = sum([basis * ctrl_pnt[2] for basis, ctrl_pnt in zip(bases, ctrl_pnts)])
        P_ts.append((P_xt, P_yt, P_zt))
        
    return P_ts

##################################
# Plotting B-Splines
##################################
def plot_bspline_2d(P_ts, ctrl_pnts, plot_ctrl=False):
    x_curve = [coord[0] for coord in P_ts]
    y_curve = [coord[1] for coord in P_ts]

    x_ctrl = [coord[0] for coord in ctrl_pnts]
    y_ctrl = [coord[1] for coord in ctrl_pnts]

    plt.plot(x_curve, y_curve, "r")
    if plot_ctrl:
        plt.plot(x_ctrl, y_ctrl)
        plt.scatter(x_ctrl, y_ctrl)
    plt.show()

def plot_bspline_3d(P_ts, ctrl_pnts, plot_ctrl=False):
    #create a figure and an axes object.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #define values to plot
    x_curve = [coord[0] for coord in P_ts]
    y_curve = [coord[1] for coord in P_ts]
    z_curve = [coord[2] for coord in P_ts]

    x_ctrl = [coord[0] for coord in ctrl_pnts]
    y_ctrl = [coord[1] for coord in ctrl_pnts]
    z_ctrl = [coord[2] for coord in ctrl_pnts]
    
    #plot the surface
    ax.plot(x_curve, y_curve, z_curve, color='r')
    
    #Plot control points
    if plot_ctrl:
        ax.plot(x_ctrl, y_ctrl, z_ctrl, color='c')
        ax.scatter(x_ctrl, y_ctrl, z_ctrl, color='c')

    #Fix axes
    ax.set_xlim([-5, 10])
    ax.set_ylim([-5, 10])
    ax.set_zlim([-5, 10])

    plt.show()


if __name__ == "__main__":
    k = 4

    ###Plot in 2d
    ctrl_pnts = [(0, 3), (1, 3), (1, 2), (2, 2), (3, 3), (4, 3), (6, 3), (8, 4), (7, 5), (8, 6)]
    n = len(ctrl_pnts) - 1
    knots = list(range(k + n + 1))
    t_vals = np.linspace(knots[3], knots[-4], 50) # For math reason leave the first three and last three intervals out.

    bspline = make_bspline_2d(t_vals, ctrl_pnts, k, knots)
    plot_bspline_2d(bspline, ctrl_pnts, plot_ctrl=False)

    ###Plot in 3d
    ctrl_pnts_3d = [(-2, 4, 1),(0, 3, 1), (1, 3, 1), (1, 2, 2), (2, 2, 2), (3, 3, 3), (4, 3, 2), (6, 3, 2), (8, 4, 2), (7, 5, 1), (8, 6, 1), (8, 7, 1)]
    n = len(ctrl_pnts_3d) - 1
    knots = list(range(k + n + 1))
    t_vals = np.linspace(knots[3], knots[-4], 50) # For math reason leave the first three and last three intervals out.

    bspline = make_bspline_3d(t_vals, ctrl_pnts_3d, k, knots)
    plot_bspline_3d(bspline, ctrl_pnts_3d, plot_ctrl=False)