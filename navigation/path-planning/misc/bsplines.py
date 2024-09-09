#!/bin/usr/python3

"""
Functions for creating B-Splines in 2D and 3D given the control points.
TODO: 1. Make the bsplines function return the position and all of its derivatives.
      2. Try to use that with a simple optimization function.
      3. Consider using a bspline library or writing a function to get the actual basis derivatives. 

The first value of vel, acc, and jerk scales with a factor of 1/dt, 1/dt^2, 1/dt^3 respectively because it's
the zero to v, a, j change. It pretty much depends on how much time we tell the system the UAV took to get to the
corresponding v, a, j value. The options are either to get rid of those values, to make them constraints in the optimization,
or to leave it as an implementation detail (temporary fix). 
"""

import numpy as np
import matplotlib.pyplot as plt

#####################################
# Make Derivatives
#####################################
def derivatives3d(traj, dt):
    """
    Computes approximate derivatives of bspline.
    """
    delta_t, prev, d_traj = dt, traj[0], [(0, 0, 0)] 
    for i in range(1, len(traj)):
        elem = traj[i]
        new_elem = ((elem[0] - prev[0]) / delta_t, (elem[1] - prev[1]) / delta_t, (elem[2] - prev[2]) / delta_t) 
        d_traj.append(new_elem)
        prev = elem
    
    return d_traj

#####################################
# Make Clamped Uniform Knot Vector
#####################################
def make_knots(t_start, t_goal, deg, num_seg):
    """
    Makes clamped uniform knots
    """
    p = deg
    m = num_seg + 2 * p
    delta_t = (t_goal - t_start) / num_seg

    knots1 = [t_start - (p - i + 1) for i in range(p + 1)]
    knots2, t = [], t_start
    for _ in range(p + 1, m - p):
        knots2.append(t)
        t += delta_t
    knots3 = [t_goal + i for i in range(p + 1)]

    return knots1 + knots2 + knots3

#####################################
# Make the B-Splines
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

def plot_bspline_3d(P_ts, V_ts, A_ts, J_ts, ctrl_pnts, plot_what=[True, True, False, False, False]):
    #create a figure and an axes object.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #define values to plot
    x_curve = [coord[0] for coord in P_ts]
    y_curve = [coord[1] for coord in P_ts]
    z_curve = [coord[2] for coord in P_ts]

    #define vel vals
    vx_curve = [coord[0] for coord in V_ts]
    vy_curve = [coord[1] for coord in V_ts]
    vz_curve = [coord[2] for coord in V_ts]

    #define acc vals
    ax_curve = [coord[0] for coord in A_ts]
    ay_curve = [coord[1] for coord in A_ts]
    az_curve = [coord[2] for coord in A_ts]

    #define jerk vals
    jx_curve = [coord[0] for coord in J_ts]
    jy_curve = [coord[1] for coord in J_ts]
    jz_curve = [coord[2] for coord in J_ts]

    #define ctrl points
    x_ctrl = [coord[0] for coord in ctrl_pnts]
    y_ctrl = [coord[1] for coord in ctrl_pnts]
    z_ctrl = [coord[2] for coord in ctrl_pnts]
    
    #plot the surface
    if plot_what[0]:
        ax.plot(x_curve, y_curve, z_curve, color='r')
    if plot_what[1]:
        ax.plot(vx_curve, vy_curve, vz_curve, color='g')
    if plot_what[2]:
        ax.plot(ax_curve, ay_curve, az_curve, color='b')
    if plot_what[3]:
        ax.plot(jx_curve, jy_curve, jz_curve, color='m')
    if plot_what[4]:
        ax.plot(x_ctrl, y_ctrl, z_ctrl, color='c')
        ax.scatter(x_ctrl, y_ctrl, z_ctrl, color='c')

    #Fix axes
    ax.set_xlim([-5, 10])
    ax.set_ylim([-5, 10])
    ax.set_zlim([-5, 10])
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    plt.show()


if __name__ == "__main__":
    k = 4
    t_start, t_goal, dt = 0, 10, 0.1 

    ###Plot in 3d
    ctrl_pnts_3d = [(-2, 4, 1),(0, 3, 1), (1, 3, 1), (1, 2, 2), (2, 2, 2), (3, 3, 3), (4, 3, 2), (6, 3, 2), (8, 4, 2), (7, 5, 1), (8, 6, 1), (8, 7, 1)]
    n = len(ctrl_pnts_3d) - 1
    num_seg = n - k + 2
    knots = make_knots(t_start, t_goal, k - 1, num_seg)
    t_vals = np.arange(knots[3], knots[-4], dt) # Only generate samples inside the non-clamped intervals.

    #Make spline and derivatives 
    bspline = make_bspline_3d(t_vals, ctrl_pnts_3d, k, knots)
    velocity = derivatives3d(bspline, dt)
    acceleration = derivatives3d(velocity, dt)
    jerk = derivatives3d(acceleration, dt)

    to_plot = [True, True, True, False, False] #Plot pos and vel
    plot_bspline_3d(bspline, velocity, acceleration, jerk, ctrl_pnts_3d, plot_what=to_plot)

