#!/bin/usr/python3

"""
Clamping works better with these splines (no division by zero error).
TODO: Move functions from bspline.py here. You can later figure out how to do clamping in your custom splines.
"""
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


# Define control points in 3D
control_points = [
    (0, 1, 0),
    (1, 2, 0),
    (2, 3, 1),
    (3, 5, 0),
    (4, 4, 1),
    (5, 2, 0)
]

def make_bspline_new(t_vals, control_points, k, knots):

    control_points = np.array([control_points])[0]
    print(control_points.shape)
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

k = 4
degree = k - 1
n = len(control_points) - 1
num_seg = n - k + 2

# Create a uniform knot vector
# knot_vector = np.concatenate(([0] * degree, np.linspace(0, 1, n - degree + 1), [1] * degree))
knot_vector = np.array(make_knots(0, 1, degree, num_seg))
t_vals = np.linspace(0, 1, 100)
print(make_bspline_new(t_vals, control_points, k, knot_vector))
# Plot the control points and B-spline curve
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot control points
# ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro--', label='Control Points')

# # Plot B-spline curve
# ax.plot(x_vals, y_vals, z_vals, 'b-', label='B-spline Curve')

# ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()
