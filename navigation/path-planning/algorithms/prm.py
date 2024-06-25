#!/bin/usr/python3 

import numpy as np
from sklearn.neighbors import KDTree 
import matplotlib.pyplot as plt 
import networkx as nx 
from dijkstras import astar, path_to_network, heuristic
from utils import point_in_collision, segment_in_collision, shortcutting, Map

######################
###Helper Functions###
######################
def build_adjacency_list(e):
    adjacency = {}
    for pnt, neighbor in e:
        adjacency[pnt] = adjacency.get(pnt, []) + [(neighbor, heuristic(pnt, neighbor))]
    
    return adjacency

def get_nearest_start_and_goal(start, goal, v):
    tree = KDTree(v, leaf_size=2)
    return (nearest_neighbors(start, v, tree, 1)[0], nearest_neighbors(goal, v, tree, 1)[0])

####################
###Main Functions###
####################
def get_random_point(size):
    x, y = np.random.rand(2) * size
    return (x, y)

def nearest_neighbors(pnt, pnts, tree, k):
    """
    Gets n nearest neighbors of pnt from a KDTree object.
    """
    pnts = np.array(pnts)
    _, indices = tree.query([pnt], k=k+1)
    neighbors = pnts[indices][0, 1:,:]
    return [tuple(neighbor) for neighbor in neighbors]

def build_prm(iters, size, obstacles, k):
    v, e = [], []
    for _ in range(iters):
        pnt = get_random_point(size)
        while point_in_collision(pnt, obstacles):
            pnt = get_random_point(size)
        v.append(pnt)

    tree = KDTree(v, leaf_size=2)
    for pnt in v: 
        for neighbor in nearest_neighbors(pnt, v, tree, k):
            edge = [pnt, neighbor]
            if not segment_in_collision(edge, obstacles):
                e.append(edge)
            
    return v, e

##############
###Plotting###
##############
def convert_to_nx(v, e):
    graph = nx.Graph()
    for vertex in v:
        graph.add_node(vertex)
    for edge in e:
        graph.add_edge(*edge)
    
    pos = {node : node for node in graph.nodes}
    return graph, pos

def plot(v, e, path, obstacles):
    nx_graph, pos = convert_to_nx(v, e)
    
    plt.title('Probabilistic Roadmap')
    nx.draw(nx_graph, pos, node_size=10, node_color="red", edge_color="orange")

    #plot obstacles
    for obstacle in obstacles: 
        obstacle.append(obstacle[0]) #Complete polygon plot 
        x, y = [], []
        for segment in obstacle:
            x.append(segment[0])
            y.append(segment[1])
        plt.plot(x, y, color="black")

    #Plot environment boundary
    plt.plot ([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color="black")

    #Plot path
    nx_path = path_to_network(path)
    nx.draw(nx_path, {node : node for node in nx_path.nodes}, node_color="green", edge_color="green", node_size=25, width=3)

    plt.show()


if __name__ == '__main__': 
    #Make map
    map = Map()
    size = map.get_size()
    obstacles = map.get_obstacles()

    #PRM parameters
    num_pnts = 500
    k = 15

    #Build PRM
    vertices, edges = build_prm(num_pnts, size, obstacles, k)
    adjacency_list = build_adjacency_list(edges)
 
    #Start and goal
    start, goal = (5, 5), (90, 90)
    nrst_start, nrst_goal = get_nearest_start_and_goal(start, goal, vertices)

    #Search best path in PRM
    distance, path, visited = astar(adjacency_list, nrst_start, nrst_goal)
    path = shortcutting([start] + path + [goal], obstacles)
    
    plot(vertices, edges, path, obstacles)


    

