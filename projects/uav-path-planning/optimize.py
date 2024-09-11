#!/bin/usr/python3

"""
Implemented spherical obstacles!!

Tried to implement a box obstacle at the origin. You can't do it with inequalities because
the inequalities devide the space into halfspaces until no trajectory can be made unless inside
the box (oposite of what we want). 

#TODO: 1. Find a way to optimize for time (the duration of traj is currently user defined, t_goal - t_start) instead of distance.
       2. Implement convex hull obstacle avoidance instead of checking every point along the trajectory.    
"""

from bsplines import make_traj, plot_bspline_3d
import numpy as np 
from scipy.optimize import minimize

#global variables
t_start, t_goal, step = 0, 2, 0.1
T_VALS = np.arange(t_start, t_goal, step) 

###############################################
# Helper Functions 
###############################################
#Unflatten control_points
def reshape_ctrl_pnts(control_points):
    """
    scipy flattens the (n,3) control_points array to (3n,). This function shapes it back to (n,3).
    """
    n_points = len(control_points) // 3
    return control_points.reshape(((n_points, 3)))

###############################################
# Objective Function
###############################################
def objective(control_points):
    """
    Computes objective function, J = j^2 + (p_goal - p_start)^2
    """
    control_points = reshape_ctrl_pnts(control_points)
    
    pos, _, _, jerk = make_traj(control_points, T_VALS)

    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    return np.sum(jerk_magnitude ** 2) + (pos[-1] - pos[0]).T @ (pos[-1] - pos[0])

##############################################
# Constraint Functions
##############################################
#Define velocity and acceleration constraints
def velocity_constraint(control_points, max_vel):
    control_points = reshape_ctrl_pnts(control_points)

    _, velocities, _, _ = make_traj(control_points, T_VALS)
    vel_magnitude = np.linalg.norm(velocities, axis=1)

    return max_vel - np.max(vel_magnitude) #Ensure largest vel is less than max_vel

def acceleration_constraint(control_points, max_acc):
    control_points = reshape_ctrl_pnts(control_points)

    _, _, accelerations, _ = make_traj(control_points, T_VALS)
    acc_magnitude = np.linalg.norm(accelerations, axis=1)

    return max_acc - np.max(acc_magnitude) #Ensure largest acc is less than max_vel

#Define initial and final condition constraints 
def constraint_pos_start(control_points, pos_start):
    control_points = reshape_ctrl_pnts(control_points)

    pos, _, _, _ = make_traj(control_points, T_VALS)
    return np.linalg.norm(pos[0] - pos_start)

def constraint_pos_goal(control_points, pos_goal):
    control_points = reshape_ctrl_pnts(control_points)

    pos, _, _, _ = make_traj(control_points, T_VALS)
    return np.linalg.norm(pos[-1] - pos_goal)

#Define obstacle constraints
def constraint_obs(control_points, obs):
    control_points = reshape_ctrl_pnts(control_points)

    obs_pos, obs_radius = obs[0], obs[1] + 1
    pos, _, _, _ = make_traj(control_points, T_VALS)
    dist_to_obs = np.sqrt((pos[:, 0] - obs_pos[0])**2 + (pos[:, 1] - obs_pos[1])**2 + (pos[:, 2] - obs_pos[2])**2)
    return dist_to_obs - obs_radius



if __name__ == '__main__':
    #Define initial gues
    initial_guess = [(-2, 4, 1),(0, 3, 1), (1, 3, 1), (1, 2, 2), (2, 2, 2), (3, 3, 3), (4, 3, 2), (6, 3, 2), (8, 4, 2), (7, 5, 1), (8, 6, 1), (8, 7, 1)]

    #Define vel and acc limits
    max_vel = 5.0
    max_acc = 3.0

    #Define initial and final conditions
    pos_start = np.array([-5, -5, -5]) 
    pos_goal = np.array([5, 5, 5])

    #Define obstacle positions and radius
    obs = [[(0, 0, 0), 3], [(-4, 0, 4), 3], [(-5, -5, -3), 1], [(-4, -4, -4), 1]]

    #Define constraints
    constraints = [
    {'type': 'ineq', 'fun': velocity_constraint, 'args': (max_vel,)},
    {'type': 'ineq', 'fun': acceleration_constraint, 'args': (max_acc,)},
    {'type': 'eq', 'fun': constraint_pos_start, 'args': (pos_start,)},
    {'type': 'eq', 'fun': constraint_pos_goal, 'args': (pos_goal,)},
    {'type': 'ineq', 'fun': constraint_obs, 'args': (obs[0],)}
    ]

    #Optimize control points to minimize jerk
    result = minimize(objective, initial_guess, constraints=constraints)

    #Reshape control points
    opt_control_points = reshape_ctrl_pnts(result.x)
    
    #Print the optimal control points
    print("\nOptimal control points:", opt_control_points)
    print("\nMinimum jerk value:", result.fun)

    #Make traj
    pos, vel, acc, jerk = make_traj(opt_control_points, T_VALS)

    print(f"\n Position traj: {pos} \n")

    #Plot traj
    plot_bspline_3d(pos, vel, acc, jerk, opt_control_points, plot_what=[True, True, True, True, False], obs=obs)