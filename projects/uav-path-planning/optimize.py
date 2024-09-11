#!/bin/usr/python3

"""
Implemented spherical obstacles!!

It works well with 2 or 3 obstacles. Obstacle constraints are non-linear as we use euclidean distance, so initial guess matters now
and you don't have an initial guess module. Anything over 2 obstacle constraints and the solver starts getting stuck in local minima.

Tried to implement a box obstacle at the origin. You can't do it with inequalities because
the inequalities devide the space into halfspaces until no trajectory can be made unless inside
the box (oposite of what we want). 

TODO: 1. Implement initial guess module. 
      2. Find a way to optimize for time (the duration of traj is currently user defined, t_goal - t_start) instead of distance.
      3. Implement convex hull obstacle avoidance instead of checking every point along the trajectory.    
      4. Get rid of for loop in constraint_obs(). Use numpy, you're not a child anymore...  
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
    
    inputs: 
        - control_points: The control points of a trajectory as a (3n,) shaped numpy array.
    outputs: 
        - The same control points but now properly stored as a (n, 3) shaped numpy array. 
    """
    n_points = len(control_points) // 3
    return control_points.reshape(((n_points, 3)))

###############################################
# Objective Function
###############################################
def objective(control_points):
    """
    Computes objective function, J = j^2

    inputs: 
        - control_points: The control points of a trajectory as (3n,) shaped numpy array.

    outputs:
        - The scalar output of the objective,
    """
    control_points = reshape_ctrl_pnts(control_points)

    _, _, _, jerk = make_traj(control_points, T_VALS)

    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    return np.sum(jerk_magnitude ** 2)

##############################################
# Constraint Functions
##############################################
#Define velocity and acceleration constraints
def velocity_constraint(control_points, max_vel):
    """
    Function used by scipy to enforce INEQUALITY constraint v(t) <= max_vel. Does this by taking the l1 norm along 
    axis 1 of the vel array (|x| + |y| + |z| for every entry) flattening the (n, 3) vel
    array to (n,). Then takes the maximum element of that array, max_elem, and returns max_vel - max_elem.
    The return format for scipy inequalities is max_vel - v(t) >= 0.

    inputs: 
        - control_points: Control points of trajectory as flattened (3n,) numpy array.
        - max_vel: The maximum velocity allowed - a scalar.

    outputs: 
        - The difference between the velocity limit and the largest velocity along the trajectory, max_vel - max(v(t))
    """
    control_points = reshape_ctrl_pnts(control_points)

    _, velocities, _, _ = make_traj(control_points, T_VALS)
    vel_magnitude = np.linalg.norm(velocities, axis=1)

    return max_vel - np.max(vel_magnitude) #Ensure largest vel is less than max_vel

def acceleration_constraint(control_points, max_acc):
    """
    Same as velocity constraint but for acceleration.
    """
    control_points = reshape_ctrl_pnts(control_points)

    _, _, accelerations, _ = make_traj(control_points, T_VALS)
    acc_magnitude = np.linalg.norm(accelerations, axis=1)

    return max_acc - np.max(acc_magnitude) #Ensure largest acc is less than max_vel

#Define initial and final condition constraints 
def constraint_pos_start(control_points, pos_start):
    """
    Function used by scipy to enforce EQUALITY constraint p(t) = pos_start.

    inputs: 
        - control_points: Control points as flattened (3n,) numpy array.
        - pos_start: Desired initial position, a (3,) numpy array.

    outputs:
        - The l1 norm of the difference between the desired and computed inital positions, a scalar.
    """
    control_points = reshape_ctrl_pnts(control_points)

    pos, _, _, _ = make_traj(control_points, T_VALS)
    return np.linalg.norm(pos[0] - pos_start)

def constraint_pos_goal(control_points, pos_goal):
    """
    Same as contraint_pos_start but for goal position.
    """
    control_points = reshape_ctrl_pnts(control_points)

    pos, _, _, _ = make_traj(control_points, T_VALS)
    return np.linalg.norm(pos[-1] - pos_goal)

#Define obstacle constraints
def constraint_obs(control_points, obs, off=0.5):
    """
    Function used by scipy to enforce INEQUALITY constraint dist_to_obs >= obs_radius. This constraints the position trajectory
    to be outside of the obstacle spheres defined by obs. This function is run for every elem in obs.

    inputs:
        - control_points: Control points as flattened (3n,) numpy array.
        - obs: a list of [pos, radius] lists, defining the center and radius of each obstacle sphere.

    outputs: 
        - The difference between the distance to the obstacle and the obstacle radius for every point on the trajectory, a (num_points,) numpy array.
    """
    control_points = reshape_ctrl_pnts(control_points)
    pos, _, _, _ = make_traj(control_points, T_VALS)

    ineqs = np.array([])
    for ob in obs:
        obs_pos, obs_radius = ob[0], ob[1]
        dist_to_obs = np.sqrt((pos[:, 0] - obs_pos[0])**2 + (pos[:, 1] - obs_pos[1])**2 + (pos[:, 2] - obs_pos[2])**2)
        diff = dist_to_obs - obs_radius - off
        ineqs = np.hstack((ineqs, diff))
        
    return ineqs



if __name__ == '__main__':
    #Define initial gues
    initial_guess = np.array([(-2, 4, 1), (0, 3, 1), (1, 3, 1), (1, 2, 2), (2, 2, 2), (3, 3, 3), (4, 3, 2), (6, 3, 2), (8, 4, 2), (7, 5, 1), (8, 6, 1), (8, 7, 1)])
    initial_guess = initial_guess.reshape(((3 * initial_guess.shape[0],)))
    
    #Define vel and acc limits
    max_vel = 5.0
    max_acc = 3.0

    #Define initial and final conditions
    pos_start = np.array([6, -4, -5]) 
    pos_goal = np.array([5 , 5, 5])

    #Define obstacle positions and radius
    obs = [[(2,  -2,   -4), 3], [(0, 0, 0), 4], [(6, -1.53743851, -1.45810787), 2]]

    #Define constraints
    constraints = [
    {'type': 'ineq', 'fun': velocity_constraint, 'args': (max_vel,)},
    {'type': 'ineq', 'fun': acceleration_constraint, 'args': (max_acc,)},
    {'type': 'eq', 'fun': constraint_pos_start, 'args': (pos_start,)},
    {'type': 'eq', 'fun': constraint_pos_goal, 'args': (pos_goal,)},
    {'type': 'ineq', 'fun': constraint_obs, 'args': (obs,)}
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