#!/bin/usr/python3 

import numpy as np
from sklearn.neighbors import KDTree 
from shapely.geometry import Point, LineString, Polygon 

def dist_func(pnt1, pnt2):
    return ((pnt2[0] - pnt1[0])**2 + (pnt2[1] - pnt1[2])**2)**0.5

def get_random_point(size):
    return np.random.rand(2) * size

def nearest_neighbors(pnt, pnts, tree, n):
    """
    Gets n nearest neighbors of pnt from a KDTree object.
    """
    _, indices = tree.query([pnt], k=n)
    return pnts[indices][:,1:,:]

def point_in_collision(pnt, obstacles):
    point, collision = Point(pnt), False
    for obstacle in obstacles: 
        polygon = Polygon(obstacle)
        collision = collision or polygon.contains(point)
    return collision 


def segment_in_collision(segment, obstacles):
    #TODO: Implement this. 
    pass

def build_prm(k, grid_size, obstacles, n=5):
    #TODO: Change numpy arrays to lists
    v, e = np.array([]), np.array([])
    for _ in range(k):
        pnt = get_random_point(grid_size)
        while point_in_collision(pnt, obstacles):
            pnt = get_random_point(grid_size)
        np.vstack((v, pnt))

    tree = KDTree(v, leaf_size=2)
    for pnt in v: 
        for neighbor in nearest_neighbors(pnt, v, tree, n):
                edge = (pnt, neighbor)
                if not segment_in_collision(edge, obstacles):
                    np.vstack((e, edge))
    
    return v, e
    





if __name__ == '__main__': 
    points = np.array([(i, j) for i, j in zip(range(10),range(10,20))])
    print(points)
    tree = KDTree(points, leaf_size=2)

    query_point = (1, 11)
    nn = nearest_neighbors(query_point, points, tree, 10)
    print(nn)
