#!/bin/usr/python3 

import heapq
import networkx as nx
import matplotlib.pyplot as plt

#TODO: Use heap to implement priority queue and speed up the algorithm by a factor of n. 
    
def make_grid(size):
    graph = {}
    cost, diag_cost = 1, (2)**0.5
    for i in range(size):
        for j in range(size):    
            if i == 0 and j ==0:
                graph[(i, j)] = [[(i,j+1),cost], [(i+1,j),cost], [(i+1,j+1), diag_cost]]  
            elif i == size-1 and j == size-1:
                graph[(i,j)] = [[(i,j-1),cost], [(i-1,j),cost], [(i-1,j-1), diag_cost]]  
            elif i == 0 and j == size-1:
                graph[(i,j)] = [[(i,j-1),cost], [(i+1,j),cost], [(i+1,j-1), diag_cost]]  
            elif i == size-1 and j ==0:
                graph[(i,j)] = [[(i,j+1),cost], [(i-1,j),cost], [(i-1,j+1), diag_cost]]  
            elif i == 0:
                graph[(i,j)] = [[(i,j+1),cost], [(i+1,j),cost], [(i, j-1),cost], [(i+1,j-1), diag_cost], [(i+1,j+1), diag_cost]]  
            elif j == 0:
                graph[(i,j)] = [[(i,j+1),cost], [(i+1,j),cost], [(i-1, j),cost], [(i-1,j+1), diag_cost], [(i+1,j+1), diag_cost]] 
            elif i == size -1:
                graph[(i,j)] = [[(i,j+1),cost], [(i-1,j),cost], [(i, j-1),cost], [(i-1,j+1), diag_cost], [(i-1,j-1), diag_cost]] 
            elif j == size -1:
                graph[(i,j)] = [[(i+1,j),cost], [(i-1,j),cost], [(i, j-1),cost], [(i+1,j-1), diag_cost], [(i-1,j-1), diag_cost]] 
            else:
                graph[(i,j)] = [[(i+1,j),cost], [(i,j+1),cost], [(i-1, j),cost], [(i,j-1), diag_cost], [(i+1,j+1), diag_cost], [(i-1,j-1), diag_cost], [(i+1,j-1), diag_cost], [(i-1,j+1), diag_cost]]  
    
    return graph
    

def grid_to_network(graph):
    net = nx.Graph()
    for node, children in graph.items():
        net.add_node(node)
        for child in children:
            child = child[0]
            net.add_node(child)
            net.add_edge(node, child)
    return net

def path_to_network(path):
    net = nx.Graph()
    
    prev = path[0]
    net.add_node(prev)
    for point in path[1:]:
        net.add_node(point)
        net.add_edge(prev, point)
        prev = point

    return net

def dijkstras(g, start, goal):
    """
    Standard dijkstra's algorithm.

    Input(s): The graph containing the edge costs, g.
              The start and goal coordinates, start, goal.
    Output(s): The optimal path and its cost as a tuple.
    """

    queue, visited = [([start], 0)], set()
    while queue: 

        #Get path of least cost
        queue = sorted(queue, key=lambda x : x[1])
        path, distance = queue.pop(0)
        
        #Get last element in path
        curr = path[-1]

        if curr == goal:
            return (path, distance, visited)
        
        #Update costs of children
        for child, edge_distance in g[curr]:
            
            if child in visited:
                continue

            new_distance = distance + edge_distance
            new_path = path + [child] 
            
            queue.append((new_path, new_distance))

        visited.add(curr)

    return None

def plot_path(grid_network, path_network, visited_network):
    pos = {node: node for node in grid_network.nodes() }
    nx.draw(grid_network, pos, node_color="gray", edge_color="gray")
    pos = {node: node for node in visited_network.nodes()}
    nx.draw(visited_network, pos, node_color="red", edge_color="none")
    pos = {node: node for node in path_network.nodes()} 
    nx.draw(path_network, pos, node_color="green", edge_color="green")
    plt.show()
    
if __name__ == '__main__':
    #Initialize grid with an "obstacle" between (2,2) and (3,3)
    graph = make_grid(10)
    graph[(2, 2)] = [[(2,3), 1], [(3,2), 1], [(3,3), float('inf')]]
    start, goal = (0,0), (7,7)

    #Run dijkstra's algorithm
    path, distance, visited = dijkstras(graph, start, goal)
    
    #Make networkx graphs for plotting
    net = grid_to_network(graph)
    pathnet = path_to_network(path)
    visitnet = path_to_network(list(visited))

    plot_path(net, pathnet, visitnet)

    

    