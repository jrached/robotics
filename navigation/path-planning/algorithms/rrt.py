#!/bin/usr/python3 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import shortcutting, get_segment_intersection

#TODO: 1. Use kd-trees for nearest neighbors instead of brute force (O(logn) vs O(n)) so that 
#         the runtime of the entire algorithms is reduced to O(nlogn).

####################
###Main Functions###
####################
def distance_func(pnt1, pnt2):
    return ((pnt2[0] - pnt1[0])**2 + (pnt2[1] - pnt1[1])**2)**0.5

def get_rand_point(grid_size):
    rand_xy = np.random.rand(2) * grid_size
    return (rand_xy[0], rand_xy[1])

def nearest_point(g, rand_pnt): #Runtime: O(n)
    return min(g.keys(), key=lambda x: distance_func(x, rand_pnt))

def get_new_node(nrst_pnt, rnd_pnt, step):
    diff_vector = tuple((a - b for a, b in zip(rnd_pnt, nrst_pnt)))
    magnitude = distance_func(nrst_pnt, rnd_pnt)
    if magnitude < step:
        step = magnitude
    new_node = tuple((a + x * (step/magnitude) for x, a in zip(diff_vector, nrst_pnt)))

    return new_node
  
def rrt(g, netx, num_iters, step, grid_size, obstacles, goal_node, eps): #Runtime O(n^2)
    """ 
    Standard rrt algorithm.

    Input(s): A dictionary containing the starting node, g. A networkx object, netx.
                The obstacles as an array containing arrays of line segments, obstacles.
                A goal, goal_node. And rrt parameters. 
    Output(s): A dictionary mapping all nodes to their immidiate parent, g.
                A networkx graph object for plotting the tree, netx. And an array
                containing all the nodes that are within a distance of the goal, paths.
    """
    paths = []
    for _ in range(num_iters):
        random_point = get_rand_point(grid_size)
        nearest = nearest_point(g, random_point)
        new_node = get_new_node(nearest, random_point, step)

        #Check for collisions 
        new_node = get_closest_collision(obstacles, nearest, new_node)

        if new_node != nearest:
            #Update graph
            g[new_node] = nearest

            #Upgdate networkx graph
            netx.add_node(new_node)
            netx.add_edge(nearest, new_node)

            #Check if goal reached
            if distance_func(new_node, goal_node) < eps:
                paths.append(new_node)

    return g, netx, paths

def reconstruct_path(g, paths):
    """
    For each node that is within a distance of the goal, end_node,
    it goes back up its parents until it reaches the starting node.

    Input(s): A dictionary mapping all nodes to their immidiate parent, g, 
                and an array containing all the end nodes, paths.
    Outputs(s): An array containing the shortest path found.
    """
    full_paths = []
    for end_node in paths:
        full_path = []
        while end_node: 
            full_path.append(end_node)
            end_node = g[end_node]
        full_paths.append(full_path)

    return full_paths if full_paths == [] else min(full_paths, key=lambda x: len(x))[::-1]

#########################
###Collision Functions###
#########################
def get_obstacle_intersection(obstacle, nearest, new_node):
    """
    Using get_segment_intersection(), it finds the intersection between all segments 
    of an obstacle and the line formed by the nearest node on the tree and the new node.
    
    Input(s): A polygon represented as a list of the points at the corners of the polygon, obstacle.
              The nearest point on the tree as a tuple, nearest.
              The point we are extending the tree towards as a tuple, new_node.
    Output(s): If there is any, the intersection of the line and the polygon segment closest to the nearest
               point on the tree. Else, the point we're extending towards.
    """
    start, intersects = obstacle[0], []
    obstacle = obstacle[1:] + [obstacle[0]]
    for end in obstacle:
        intersect = get_segment_intersection([nearest, new_node], [start, end])
        start = end
        if intersect:
            intersects.append(intersect)
    
    return new_node if len(intersects) == 0 else min(intersects, key=lambda x : distance_func(x, nearest)) 

def get_closest_collision(obstacles, nearest, new_node):
    """
    Get the closest intersection between any of the line segments 
    of any of the obstacles in the field and the line formed by the 
    nearest point on the tree and the point we're extending the tree towards.

    Input(s): A list of polygons, obstacles.
              The nearest point on the tree as a tuple, nearest.
              The point we are extending the tree towards as a tuple, new_node.
    Output(s): The closest intersection out of all the obstacles and our line of interest.
               If there isn't any, the output will be new_node.               
    """
    collisions = []
    for obstacle in obstacles: 
        collision = get_obstacle_intersection(obstacle, nearest, new_node)
        collisions.append(collision)
    
    cc = min(collisions, key=lambda x: distance_func(x, nearest))
    if cc != new_node: #Avoid python rounding error at edges of obstacle.
        mag = distance_func(nearest, cc)
        cc = (nearest[0] + 0.01 * (nearest[0] - cc[0]) / mag, nearest[1] + 0.01 * (nearest[1] - cc[1]) / mag)
    return cc

##############
###Plotting###
############## 
def plot_bbox(pnt, goal=False):
    color = "red" if goal else "green"
    x, y = pnt[0], pnt[1]
    bbox = [[(x-2, y-2), (x-2, y+2)], [(x-2, y+2), (x+2, y+2)], [(x+2, y+2), (x+2, y-2)], [(x+2, y-2), (x-2, y-2)]]
    for segment in bbox:
        start, end = segment[0], segment[1]
        plt.plot((start[0], end[0]), (start[1], end[1]), color = color)
    
def plot_graph(g, grid_size, obstacles, origin, goal, path):
       
    #Plot tree
    plt.title('Rapidly-exploring Random Tree (RRT)')
    nx.draw(g, {node: node for node in g.nodes()}, with_labels=False, node_size=0, node_color='blue', edge_color='grey')
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)

    #Plot obstacles 
    for obstacle in obstacles:
        start = obstacle[0]
        obstacle = obstacle + [obstacle[0]]
        for end in obstacle:
            plt.plot((start[0], end[0]), (start[1], end[1]), color = "black")
            start = end
    
    #Plot environment boundary
    plt.plot ([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color="black")

    #Plot starting and end point
    plot_bbox(origin)
    plot_bbox(goal, goal=True)

    #Plot path
    if path != []:
        start = path[0]
        for end in path[1:]:
            plt.plot((start[0], end[0]), (start[1], end[1]), color = "blue")
            start = end

    plt.show()
    

if __name__ == "__main__":
    #RRT parameters
    grid_size = 100
    step_size = 5
    iters = 3000
    eps = 3

    #Initialize graphs
    start_node = (5, 5)
    goal = (90, 90)
    graph = {start_node: None}
    netx_g = nx.Graph()
    netx_g.add_node(start_node)

    #Initialize obstacles
    obstacles = [[(85,25), (90, 30), (85, 35), (80,30), (85, 25)],
                 [(20,20), (30, 20), (30, 30), (20,30), (20, 20)], 
                 [(20,80), (35, 80), (35, 95), (20,95), (20, 80)],
                 [(60,60), (80, 60), (80, 80), (60,80), (60, 60)]]

    new_graph, new_g, paths = rrt(graph, netx_g, iters, step_size, grid_size, obstacles, goal, eps)
    best_path = reconstruct_path(new_graph, paths) 
    best_path = shortcutting(best_path + [goal], obstacles)
    
    plot_graph(new_g, grid_size, obstacles, start_node, goal, best_path)