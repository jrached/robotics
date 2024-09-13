#!/bin/usr/python3
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Functions for creating B-Splines in 3D given the control points.
TODO: 1. Get derivatives using spline library.
      2. Make all functions work with numpy. Currently make_bspline expects numpy control points, but make derivatives
         expects a python list.
"""

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

    knots1 = [t_start for i in range(p+1)]
    knots2, t = [], t_start
    for _ in range(p+1, m - p):
        t += delta_t
        knots2.append(t)
    knots3 = [t_goal for i in range(p+1)]
    
    return knots1 + knots2 + knots3

#####################################
# Make Bspline
#####################################
def make_bspline_new(t_vals, control_points, k, knots):

    control_points = np.array([control_points])[0] #Make control into numpy array 
    
    # Define a B-spline for each dimension (x, y, z)
    bspline_x = BSpline(knots, control_points[:, 0], k - 1)
    bspline_y = BSpline(knots, control_points[:, 1], k - 1)
    bspline_z = BSpline(knots, control_points[:, 2], k - 1)
   
    # Generate B-spline points along the curve
    x_vals = bspline_x(t_vals)
    y_vals = bspline_y(t_vals)
    z_vals = bspline_z(t_vals)

    position = np.vstack((x_vals, y_vals, z_vals))

    return position.T


####################################
# Make Trajectory For Optimization 
####################################
def make_traj(ctrl_pnts, t_vals, k=4):
    """
    From control points, t intervals, and order of the b-spline compute the bspline and its derivatives.
    
    inputs: 
        - ctrl_pnts: Control points of bspline.
        - t_vals: Time intervals sample along the trajecotry. 
        - k: Order of the bspline.
    
    outputs:
        - bspline: The x, y, z position values of the bspline trajectory.
        - other 3: The derivatives of bspline. 
    """
    n = len(ctrl_pnts) - 1
    num_seg = n - k + 2
    t_start, t_goal = t_vals[0], t_vals[-1]
    knots = make_knots(t_start, t_goal, k - 1, num_seg)
    
    #Make spline and derivatives 
    dt = 1 / (t_goal - t_start)
    bspline = make_bspline_new(t_vals, ctrl_pnts, k, knots)
    velocity = derivatives3d(bspline, dt)
    acceleration = derivatives3d(velocity, dt)
    jerk = derivatives3d(acceleration, dt)

    return np.array(bspline), np.array(velocity), np.array(acceleration), np.array(jerk)

##################################
# Plotting B-Splines
##################################
def plot_bspline_3d(P_ts, V_ts, A_ts, J_ts, ctrl_pnts, plot_what=[True, True, False, False, False], obs=None):
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
    legend = []
    if plot_what[0]:
        ax.plot(x_curve, y_curve, z_curve, color='r')
        legend.append("Pos")
    if plot_what[1]:
        ax.plot(vx_curve, vy_curve, vz_curve, color='g')
        legend.append("Vel")
    if plot_what[2]:
        ax.plot(ax_curve, ay_curve, az_curve, color='b')
        legend.append("Acc")
    if plot_what[3]:
        ax.plot(jx_curve, jy_curve, jz_curve, color='m')
        legend.append("Jerk")
    if plot_what[4]:
        ax.plot(x_ctrl, y_ctrl, z_ctrl, color='c')
        ax.scatter(x_ctrl, y_ctrl, z_ctrl, color='c')
        legend.append("Ctrl Pnts")

    #Plot spherical obs
    if obs:
        for c, r in obs:
            # draw sphere
            u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
            x = r*np.cos(u)*np.sin(v)
            y = r*np.sin(u)*np.sin(v)
            z = r*np.cos(v)

            ax.plot_surface(-x+c[0], -y+c[1], -z+c[2], color='y', alpha=0.2)

    #Fix axes 
    x_min, x_max = min(x_curve), max(x_curve)
    y_min, y_max = min(y_curve), max(y_curve)
    z_min, z_max = min(z_curve), max(z_curve)

    range_max, range_min = max(x_max, y_max, z_max), min(x_min, y_min, z_min)

    ax.set_xlim([range_min - 2, range_max + 2])
    ax.set_ylim([range_min - 2, range_max + 2])
    ax.set_zlim([range_min - 2, range_max + 2])
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.legend(legend)

    plt.show()


if __name__ == "__main__":
    #Make trajectory times 
    t_start, t_goal, step = 0, 10, 0.1 
    t_vals = np.arange(t_start, t_goal, step) 
    
    #Define control points
    ctrl_pnts_3d = [(-2, 4, 1),(0, 3, 1), (1, 3, 1), (1, 2, 2), (2, 2, 2), (3, 3, 3), (4, 3, 2), (6, 3, 2), (8, 4, 2), (7, 5, 1), (8, 6, 1), (8, 7, 1)]

    #Make spline and derivatives 
    traj = make_traj(ctrl_pnts_3d, t_vals, k=4)

    #Plot in 3D
    to_plot = [True, True, True, False, True] #Plot pos and vel
    plot_bspline_3d(*traj, ctrl_pnts_3d, plot_what=to_plot)

