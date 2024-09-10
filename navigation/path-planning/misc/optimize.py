#!/bin/usr/python3

"""
The optimization is being run.
It seems to be optimizing right.
The issue is that in minimizing jerk, it just 
picks all the control points in the same point in space.

I added the term -||qf - qi|| to the objective so that it would 
reward long trajectories (so that we don't have all control points 
at the same point) and it did produce a trajectory. It's just too long lol.

Maybe something along those lines can be a permanent fix.
Maybe the initial guess mitigates this problem (though I'm not convinced). 
"""

from bsplines import make_traj, plot_bspline_3d
import numpy as np 
from scipy.optimize import minimize

# Define your bspline trajectory 
#imported.

#global variables
t_start, t_goal, step = 0, 2, 0.1
t_vals = np.arange(t_start, t_goal, step) 

#define objective 
def objective(control_points):
    n_points = len(control_points) // 3
    control_points = control_points.reshape(((n_points, 3)))
    
    pos, vel, acc, jerk = make_traj(control_points, t_vals)

    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    return np.sum(jerk_magnitude ** 2) - (pos[-1, :] - pos[0, :]).T @ (pos[-1, :] - pos[0, :])

#Define vel and acc limits
max_vel = 5.0
max_acc = 3.0

#Define initial and final control points
first_point = np.array([-2, 4, 1])
last_point = np.array([8, 7, 1])

#Define velocity and acceleration constraints
def velocity_constraint(control_points):
    n_points = len(control_points) // 3
    control_points = control_points.reshape(((n_points, 3)))

    _, velocities, _, _ = make_traj(control_points, t_vals)
    vel_magnitude = np.linalg.norm(velocities, axis=1)

    return max_vel - np.max(vel_magnitude) #Ensure largest vel is less than max_vel

def acceleration_constraint(control_points):
    n_points = len(control_points) // 3
    control_points = control_points.reshape(((n_points, 3)))

    _, _, accelerations, _ = make_traj(control_points, t_vals)
    acc_magnitude = np.linalg.norm(accelerations, axis=1)

    return max_acc - np.max(acc_magnitude) #Ensure largest vel is less than max_vel

#Define initial and final condition constraints (in this case control points)
def constraint_first_point(control_points, first_point):
    control_points = control_points.reshape((-1, 3))
    return np.linalg.norm(control_points[0] - first_point)

def constraint_last_point(control_points, last_point):
    control_points = control_points.reshape((-1, 3))
    return np.linalg.norm(control_points[-1] - last_point)

#Define initial guess
initial_guess = [(-2, 4, 1),(0, 3, 1), (1, 3, 1), (1, 2, 2), (2, 2, 2), (3, 3, 3), (4, 3, 2), (6, 3, 2), (8, 4, 2), (7, 5, 1), (8, 6, 1), (8, 7, 1)]

#Define constraints
constraints = [
    {'type': 'ineq', 'fun': velocity_constraint},
    {'type': 'ineq', 'fun': acceleration_constraint},
    {'type': 'eq', 'fun': constraint_first_point, 'args': (first_point,)},
    {'type': 'eq', 'fun': constraint_last_point, 'args': (last_point,)}
]

#Optimize control points to minimize jerk
result = minimize(objective, initial_guess, constraints=constraints)

#Print the optimal control points
print("Optimal control points:", result.x)
print("Minimum jerk value:", result.fun)

#Reshape control points
n_points = len(result.x) // 3
control_points = result.x.reshape(((n_points, 3)))

#Make traj
pos, vel, acc, jerk = make_traj(control_points, t_vals)

print(f"\n This is the position traj: {pos} \n")

#Plot traj
plot_bspline_3d(pos, vel, acc, jerk, control_points, plot_what=[True, True, True, True, True])