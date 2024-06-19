#!/bin/usr/python3 

import heapq
import networkx as nx
import matplotlib.pyplot as plt

#TODO: Make a trajectory class where each trajectory has two properties: path and cost.

####################
###Priority Queue###
####################
class PriorityQueue():
    """
    Priority queue for arrays of trajectory objects.
    """

    def __init__(self, queue):
        dic, heap = {}, []
        for path, cost in queue:
            dic[cost] = dic.get(cost, []) + [path]
            heap.append(cost)
        
        self.dic = dic
        self.heap = heap
        heapq.heapify(self.heap)
    
    def add_to_queue(self, new_path):
        path, cost = new_path
        self.dic[cost] = self.dic.get(cost, []) + [path]
        heapq.heappush(self.heap, cost)

    def pop_queue(self):
        opt_cost = heapq.heappop(self.heap)
        opt_paths = self.dic[opt_cost]
        opt_path = opt_paths.pop(0)

        self.dic[opt_cost] = opt_paths
        
        return (opt_path, opt_cost)
        
#####################
###Grid Functions####
#####################        
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

##########################
###Dijkstra's Algorithm###
##########################
def dijkstras(g, start, goal):
    """
    Standard dijkstra's algorithm.

    Input(s): The graph containing the edge costs, g.
              The start and goal coordinates, start, goal.
    Output(s): The optimal path and its cost as a tuple.
    """

    pqueue, visited = PriorityQueue([([start], 0)]), set()
    while pqueue: 

        #Get path of least cost
        path, distance = pqueue.pop_queue()
        
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
            
            pqueue.add_to_queue((new_path, new_distance))

        visited.add(curr)

    return None

##############
###Plotting###
##############
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

