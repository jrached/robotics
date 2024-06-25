#!/bin/usr/python3 

from shapely.geometry import Point, LineString, Polygon 
import numpy as np

###############
###Map Class###
###############
class Map():
    def __init__(self, size=100, obstacles=None):
        self.size = size
        if not obstacles:
            obstacles = []
            for center in [(40, 40), (20, 20), (20, 80), (70, 70), (60, 10)]:
                obstacle = [(center[0]+dx, center[1]+dy) for dx, dy in [(0, 10), (10, 20), (20, 10), (10, 0)]]
                obstacles.append(obstacle)
            self.obstacles = obstacles
        
    def get_obstacles(self):
        return self.obstacles
    
    def get_size(self):
        return self.size
    
    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
    
    def set_size(self, size):
        self.size = size

#########################
###Collision Functions###
#########################
def point_in_collision(pnt, obstacles):
    """
    Uses shapely library to check if a point is in collision with different polygons.

    Input(s): A point in 2D as a tuple, pnt. A list of obstacles represented as 
              lists of the points at the corners of the polygon, obstacles.
    Output(s): Bool, wheter or not the point is in collision.
    """
    point, collision = Point(pnt), False
    for obstacle in obstacles: 
        polygon = Polygon(obstacle)
        collision = collision or polygon.contains(point)
    return collision 

def segment_in_collision(segment, obstacles):
    """
    Uses shapely library to check if a line segment is in collision with different polygons.

    Input(s): A line segment in 2D represented as a list containing the two endpoints of the 
              segment represented as tuples, segment. A list of obstacles represented as 
              lists of the points at the corners of the polygon, obstacles.
    Output(s): Bool, wheter or not the segment is in collision.
    """
    line_segment, collision = LineString(segment), False
    for obstacle in obstacles: 
        polygon = Polygon(obstacle)
        collision = collision or polygon.intersects(line_segment)
    return collision 

def get_segment_intersection(line1, line2):
    """
    Given two line segments, it returns the point at which they intersect.

    Input(s): 2 lists containing the starting and end points, as tuples, of two line segments. 
    Ouput(s): A tuple representing the point at which they intersect. None if they don't intersect.
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

##########################
###Path Post-Processing###
##########################
def shortcutting(path, obstacles, factor=5):
    """
    Randomly selects two points down the path, draws a straight line between them,
    if that segment isn't in collision with any obstacles, then it gets rid of every point 
    in between the two points and connects them with the new line.

    Input(s): A path as a list of tuples, where each tuple is a point along the path, path.
              A list of obstacles as lists of points at the corners of each polygon, obstacles.
              The number of iterations as a factor of the length of the path, factor. (optional).
    Output(s): The shorcutted path.
    """
    iters = factor * len(path)
    for _ in range(iters): 
        ndx1, ndx2 = np.sort(np.random.randint(0, len(path), 2))
        pnt1, pnt2 = path[ndx1], path[ndx2]
        if pnt1 != pnt2 and not segment_in_collision((pnt1, pnt2), obstacles):
            path = path[:ndx1+1] + path[ndx2:] 

    return path