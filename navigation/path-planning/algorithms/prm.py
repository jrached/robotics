#!/bin/usr/python3 

import numpy as np
from sklearn.neighbors import KDTree 
from shapely.geometry import Point, LineString, Polygon 
import matplotlib.pyplot as plt 
import networkx as nx 
from dijkstras import astar, path_to_network, heuristic

#TODO: Run A* on PRM and plot output.

#########################
###Collision Functions###
#########################
def point_in_collision(pnt, obstacles):
    point, collision = Point(pnt), False
    for obstacle in obstacles: 
        polygon = Polygon(obstacle)
        collision = collision or polygon.contains(point)
    return collision 

def segment_in_collision(segment, obstacles):
    line_segment, collision = LineString(segment), False
    for obstacle in obstacles: 
        polygon = Polygon(obstacle)
        collision = collision or polygon.intersects(line_segment)
    return collision 

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

def build_prm(iters, grid_size, obstacles, k):
    v, e = [], []
    for _ in range(iters):
        pnt = get_random_point(grid_size)
        while point_in_collision(pnt, obstacles):
            pnt = get_random_point(grid_size)
        v.append(pnt)

    tree = KDTree(v, leaf_size=2)
    for pnt in v: 
        for neighbor in nearest_neighbors(pnt, v, tree, k):
            edge = [pnt, neighbor]
            if not segment_in_collision(edge, obstacles):
                e.append(edge)
            
    return v, e

def build_adjacency_list(e):
    adjacency = {}
    for pnt, neighbor in e:
        adjacency[pnt] = adjacency.get(pnt, []) + [(neighbor, heuristic(pnt, neighbor))]
    
    return adjacency

def get_nearest_start_and_goal(start, goal, v):
    tree = KDTree(v, leaf_size=2)
    return (nearest_neighbors(start, v, tree, 1)[0], nearest_neighbors(goal, v, tree, 1)[0])

def shortcutting(path, obstacles, factor=5):
    new_path = path
    iters = factor * len(path)
    for _ in range(iters): 
        ndx1, ndx2 = np.sort(np.random.randint(0, len(new_path), 2))
        pnt1, pnt2 = new_path[ndx1], new_path[ndx2]
        if not segment_in_collision((pnt1, pnt2), obstacles):
            new_path = new_path[:ndx1+1] + new_path[ndx2:] 

    return new_path
    
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

def plot_prm(v, e, path, obstacles):
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
        plt.plot(x, y, color="blue")

    #Plot environment boundary
    x, y = [0, 0, 100, 100, 0], [0, 100, 100, 0, 0]
    plt.plot (x, y, color="blue")

    #Plot path
    nx_path = path_to_network(path)
    pos = {node : node for node in nx_path.nodes}
    nx.draw(nx_path, pos, node_color="green", edge_color="green", node_size=10)

    plt.show()


if __name__ == '__main__': 
    #PRM parameters
    num_pnts = 1000
    grid_size = 100
    k = 30

    #Define obstacles
    obstacles = []
    for center in [(40, 40), (20, 20), (20, 80), (70, 70), (60, 10)]:
        obstacle = [(center[0]+dx, center[1]+dy) for dx, dy in [(0, 10), (10, 20), (20, 10), (10, 0)]]
        obstacles.append(obstacle)

    #Build PRM
    vertices, edges = build_prm(num_pnts, grid_size, obstacles, k)
    adjacency_list = build_adjacency_list(edges)
    
    #Start and goal
    start, goal = (50, 95), (58, 11)
    nrst_start, nrst_goal = get_nearest_start_and_goal(start, goal, vertices)

    #Search best path in PRM
    distance, path, visited = astar(adjacency_list, nrst_start, nrst_goal)

    plot_prm(vertices, edges, path, obstacles)


    

