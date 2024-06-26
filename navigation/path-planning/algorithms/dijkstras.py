#!/bin/usr/python3 

import heapq
import networkx as nx
import matplotlib.pyplot as plt

####################
###Priority Queue###
####################
class PriorityQueue(): 
    def __init__(self, elements):
        """
        elements contains tuples (int, list), where the list in the second entry 
        is a path (which is a list of tuples). 
        """
        heapq.heapify(elements)
        self.heap = elements
    
    def add_to_queue(self, distance, path):
        heapq.heappush(self.heap, (distance, path))
    
    def pop_queue(self):
        return heapq.heappop(self.heap)[1]
        
#####################
###Grid Functions####
##################### 
def inbounds(pnt, size):
    return 0 <= pnt[0] <= size - 1 and 0 <= pnt[1] <= size - 1

def not_in_walls(pnt, walls):
    flag = True
    for wall in walls:
        ((low_x, high_x), (low_y, high_y)) = wall
        flag = flag and not (low_x <= pnt[0] <= high_x and low_y <= pnt[1] <= high_y)
    return flag

def make_grid(size, walls=[]):
    graph, cost, diag_cost = {}, 1, (2)**0.5
    for i in range(size):
        for j in range(size):
            neighbors = []    
            for dx, dy, dist in [(1,0,cost), (0,1,cost), (-1,0,cost), (0,-1,cost), (1,1,diag_cost), (-1,1,diag_cost), (1,-1,diag_cost), (-1,-1,diag_cost)]:
                neighbors.append(((i + dx, j + dy), dist))
            neighbors = list(filter(lambda x: inbounds(x[0], size), neighbors))
            neighbors = list(filter(lambda x: not_in_walls(x[0], walls), neighbors))
            graph[(i,j)] = neighbors
    return graph
   
def grid_to_network(graph):
    net = nx.Graph()
    for node, children in graph.items():
        if children:
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

##########################
###Dijkstra's Algorithm###
##########################
def dijkstras(g, start, goal):
    """
    Standard dijkstra's algorithm. Runtime: O((V + E)log(V))

    Input(s): The graph, a weighted adjacency list, g.
              The start and goal coordinates, start, goal.
    Output(s): The optimal path and its cost as a tuple.
    """

    pqueue, visited, cost_so_far = PriorityQueue([(0, [start])]), set(), {start: 0}
    while pqueue: 

        path = pqueue.pop_queue()
        curr = path[-1]

        if curr in visited: 
            continue

        if curr == goal:
            break
        
        for child, edge_distance in g[curr]:
            
            if child in visited:
                continue

            new_distance = cost_so_far[curr] + edge_distance

            if child not in cost_so_far or new_distance < cost_so_far[child]:
                new_path = path + [child] 
                cost_so_far[child] = new_distance
                pqueue.add_to_queue(new_distance, new_path)
        
        visited.add(curr)

    return (cost_so_far[curr], path, visited)

##################
###A* Algorithm###
##################
def heuristic(pnt1, pnt2, euclidean=True):
    if euclidean:
        return ((pnt2[0] - pnt1[0])**2 + (pnt2[1] - pnt1[1])**2)**0.5
    return abs(pnt2[0] - pnt1[0]) + abs(pnt2[1] - pnt1[1])

def astar(g, start, goal):
    """
    Standard A* algorithm.

    Input(s): The graph, a weighted adjacency list, g.
              The start and goal coordinates, start, goal.
    Output(s): The optimal path and its cost as a tuple.
    """

    pqueue, visited, cost_so_far = PriorityQueue([(0, [start])]), set(), {start : 0}
    while pqueue: 

        path = pqueue.pop_queue()
        curr = path[-1]

        if curr == goal:
            break 
        
        for child, edge_distance in g[curr]:
            
            if child in visited:
                continue

            new_distance = cost_so_far[curr] + edge_distance
            
            if child not in cost_so_far or new_distance < cost_so_far[child]:
                new_path = path + [child] 
                cost_so_far[child] = new_distance
                pqueue.add_to_queue(new_distance + heuristic(child, goal), new_path)
        
        visited.add(curr)

    return (cost_so_far[curr], path, visited)

##############
###Plotting###
##############
def plot_path(grid_network, path_network, visited_network):
    pos = {node: node for node in grid_network.nodes() }
    nx.draw(grid_network, pos, node_size=10, node_color="gray", edge_color="gray")
    pos = {node: node for node in visited_network.nodes()}
    nx.draw(visited_network, pos, node_size=10, node_color="red", edge_color="none")
    pos = {node: node for node in path_network.nodes()} 
    nx.draw(path_network, pos, node_size=10, node_color="green", edge_color="green")
    plt.show()


if __name__ == '__main__':
    #Initialize grid
    walls = [((40,60),(40,60)), ((60,80), (10,30)), ((20,30),(20,30))]
    grid_size = 100
    graph = make_grid(grid_size, walls)
    start, goal = (5,50), (80, 45)

    #Run dijkstra's or astar algorithm
    distance, path, visited = astar(graph, start, goal)

    # Make networkx graphs for plotting
    net = grid_to_network(graph)
    pathnet = path_to_network(path)
    visitnet = path_to_network(list(visited))

    plot_path(net, pathnet, visitnet)

