#!/bin/usr/python3

"""
The optimization works! It finds the shortest path from start to goal.

Tried to implement a box obstacle at the origin. You can't do it with inequalities because
the inequalities devide the space into halfspaces until no trajectory can be made unless inside
the box (oposite of what we want). 

#TODO: Add an obstacle at the center to see if it avoids it smoothly.

Maybe the next step is to proceed with the mader approach to obstacle avoidance and implement 
a convex hull method where you can get the hull of a segment and the plane orthogonal to it.
"""

from bsplines import make_traj, plot_bspline_3d
import numpy as np 
from scipy.optimize import minimize

#global variables
t_start, t_goal, step = 0, 2, 0.1
t_vals = np.arange(t_start, t_goal, step) 

#Unflatten control_points
def reshape_ctrl_pnts(control_points):
    n_points = len(control_points) // 3
    return control_points.reshape(((n_points, 3)))

#define objective 
def objective(control_points):
    control_points = reshape_ctrl_pnts(control_points)
    
    _, _, _, jerk = make_traj(control_points, t_vals)

    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    return np.sum(jerk_magnitude ** 2) 

#Define vel and acc limits
max_vel = 5.0
max_acc = 3.0

#Define initial and final conditions
pos_start = np.array([5, -5, -5]) 
pos_goal = np.array([5, 5, 5])

#Define velocity and acceleration constraints
def velocity_constraint(control_points):
    control_points = reshape_ctrl_pnts(control_points)

    _, velocities, _, _ = make_traj(control_points, t_vals)
    vel_magnitude = np.linalg.norm(velocities, axis=1)

    return max_vel - np.max(vel_magnitude) #Ensure largest vel is less than max_vel

def acceleration_constraint(control_points):
    control_points = reshape_ctrl_pnts(control_points)

    _, _, accelerations, _ = make_traj(control_points, t_vals)
    acc_magnitude = np.linalg.norm(accelerations, axis=1)

    return max_acc - np.max(acc_magnitude) #Ensure largest acc is less than max_vel

#Define initial and final condition constraints 
def constraint_pos_start(control_points, pos_start):
    control_points = reshape_ctrl_pnts(control_points)

    pos, _, _, _ = make_traj(control_points, t_vals)
    return np.linalg.norm(pos[0] - pos_start)

def constraint_pos_goal(control_points, pos_goal):
    control_points = reshape_ctrl_pnts(control_points)

    pos, _, _, _ = make_traj(control_points, t_vals)
    return np.linalg.norm(pos[-1] - pos_goal)


#Define initial gues
initial_guess = [(-2, 4, 1),(0, 3, 1), (1, 3, 1), (1, 2, 2), (2, 2, 2), (3, 3, 3), (4, 3, 2), (6, 3, 2), (8, 4, 2), (7, 5, 1), (8, 6, 1), (8, 7, 1)]

#Define constraints
constraints = [
    {'type': 'ineq', 'fun': velocity_constraint},
    {'type': 'ineq', 'fun': acceleration_constraint},
    {'type': 'eq', 'fun': constraint_pos_start, 'args': (pos_start,)},
    {'type': 'eq', 'fun': constraint_pos_goal, 'args': (pos_goal,)},
]


if __name__ == '__main__':
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
    plot_bspline_3d(pos, vel, acc, jerk, control_points, plot_what=[True, True, True, True, False])