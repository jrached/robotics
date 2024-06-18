#!/bin/usr/python3 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
    new_node = tuple((a + x * (step/magnitude) for x, a in zip(diff_vector, nrst_pnt)))

    return new_node
  
def rrt(g, netx, num_iters, step, grid_size, obstacles): #Runtime O(n^2)
    for _ in range(num_iters):
        random_point = get_rand_point(grid_size)
        nearest = nearest_point(g, random_point)
        new_node = get_new_node(nearest, random_point, step)

        #Check for collisions 
        new_node = get_closest_collision(obstacles, nearest, new_node)

        if new_node != nearest:
            #Update graph
            g[nearest].append(new_node)
            g[new_node] = []

            #Upgdate networkx graph
            netx.add_node(new_node)
            netx.add_edge(nearest, new_node)

    return g, netx

#########################
###Collision Functions###
#########################
def get_segment_intersection(line1, line2):
    """
    Given two line segments, it returns the point at which they intersect.

    Input(s): 2 lists containing the starting and end points of two line segments. 
    Ouput(s): A tuple representing the point at which they intersect.
    """

    x1, x2, y1, y2 = line1[0][0], line1[1][0], line1[0][1], line1[1][1]
    a1, a2, b1, b2 = line2[0][0], line2[1][0], line2[0][1], line2[1][1]
    denominator = (y2 - y1) * (a2 - a1) - (x2 - x1) * (b2 - b1) 
    numerator = (b1 - y1) * (a2 - a1) + (x1 - a1) * (b2 - b1) 

    if denominator == 0:
        return None
    
    alpha = numerator / denominator 
    
    if a2 == a1:
        beta = (y1 - b1 + alpha * (y2 - y1)) / (b2 - b1) 
    else:
        beta = (x1 - a1 + alpha * (x2 - x1)) / (a2 - a1)

    if (alpha > 1 or alpha < 0) or (beta > 1 or beta < 0): 
        return None
    
    return (x1 + alpha * (x2 - x1), y1 + alpha * (y2 - y1))
    
def get_obstacle_intersection(obstacle, nearest, new_node):
    line = [nearest, new_node]
    intersects = []
    for segment in obstacle:
        intersect = get_segment_intersection(line, segment)
        if intersect:
            intersects.append(intersect)
    
    if len(intersects) == 0:
        return new_node
    return min(intersects, key=lambda x : distance_func(x, nearest))

def get_closest_collision(obstacles, nearest, new_node):
    collisions = []
    for obstacle in obstacles: 
        collision = get_obstacle_intersection(obstacle, nearest, new_node)
        collisions.append(collision)
    return min(collisions, key=lambda x: distance_func(x, nearest))

##############
###Plotting###
############## 
def plot_graph(g, grid_size, obstacles):
    pos = {node: node for node in g.nodes()}  # positions for all nodes
    
    plt.title('Rapidly-exploring Random Tree (RRT)')
    nx.draw(g, pos, with_labels=False, node_size=5, node_color='blue', edge_color='grey')
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.axis('on')

    #Plot obstacles 
    for obstacle in obstacles:
        for segment in obstacle:
            start, end = segment[0], segment[1]
            plt.plot((start[0], end[0]), (start[1], end[1]), color = "black")

    #Plot starting point 
    for segment in [[(49, 49), (49, 51)], [(49, 51), (51, 51)], [(51, 51), (51, 49)], [(51, 49), (49, 49)]]:
        start, end = segment[0], segment[1]
        plt.plot((start[0], end[0]), (start[1], end[1]), color = "green")

    plt.show()
    

if __name__ == "__main__":
    #RRT parameters
    grid_size = 100
    step_size = 2
    iters = 2000

    #Initialize graphs
    start_node = (grid_size/2, grid_size/2)
    graph = {start_node: []}
    netx_g = nx.Graph()
    netx_g.add_node(start_node)

    #Initialize obstacles
    obstacles = [[[(20,20), (30,20)], [(30, 20), (30, 30)], [(30, 30), (20,30)], [(20,30), (20, 20)]], 
                 [[(60,60), (80,60)], [(80, 60), (80, 80)], [(80, 80), (60,80)], [(60,80), (60, 60)]]]

    new_graph, new_g = rrt(graph, netx_g, iters, step_size, grid_size, obstacles)
    plot_graph(new_g, grid_size, obstacles)

    
#Note if we implemented nearest neighbors using kd-trees, it would take O(logn) time to both insert and query
#to our graph, meaning our entire algorithm would have an O(nlogn) runtime.



